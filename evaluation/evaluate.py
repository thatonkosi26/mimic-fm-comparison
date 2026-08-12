"""
evaluation/evaluate.py

Implements Section 3.6 end to end: pulls together the val/test
predictions already saved by every model script (baselines.py, lstm.py,
tft.py, chronos_eval.py, chronos_finetune.py), computes the full set of
metrics for each of the 7 model configurations x 2 imputation
conditions, and runs the McNemar imputation-sensitivity test
(Section 3.6.3, Research Question 3) comparing forward_fill vs
linear_interp within each model.

Usage:
    python evaluation/evaluate.py

Output:
    results/evaluation/full_results.csv          -- one row per model x condition
    results/evaluation/imputation_sensitivity.csv -- one row per model (McNemar + CI overlap)
    results/evaluation/full_results_summary.md    -- human-readable summary
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, PROJECT_ROOT

from metrics import (
    best_f1_threshold, sensitivity_specificity, expected_calibration_error,
    platt_scale, bootstrap_ci, bootstrap_ci_f1, mcnemar_test,
)

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "evaluation")

# Where each model's val/test prediction .npy files live, relative to
# results/<condition>/... Matches the output paths each model script
# already writes (see each script's own docstring for confirmation).
MODEL_PRED_PATHS = {
    "logistic_regression": lambda c: (
        f"baselines/{c}/logistic_regression_val_predictions.npy",
        f"baselines/{c}/logistic_regression_test_predictions.npy",
    ),
    "random_forest": lambda c: (
        f"baselines/{c}/random_forest_val_predictions.npy",
        f"baselines/{c}/random_forest_test_predictions.npy",
    ),
    "xgboost": lambda c: (
        f"baselines/{c}/xgboost_val_predictions.npy",
        f"baselines/{c}/xgboost_test_predictions.npy",
    ),
    "lstm": lambda c: (
        f"lstm/{c}/val_predictions.npy",
        f"lstm/{c}/test_predictions.npy",
    ),
    "tft": lambda c: (
        f"tft/{c}/val_predictions.npy",
        f"tft/{c}/test_predictions.npy",
    ),
    "chronos_zeroshot": lambda c: (
        f"chronos/{c}/zeroshot_val_predictions.npy",
        f"chronos/{c}/zeroshot_test_predictions.npy",
    ),
    "chronos_finetuned": lambda c: (
        f"chronos/{c}/finetuned_val_predictions.npy",
        f"chronos/{c}/finetuned_test_predictions.npy",
    ),
}


def _load_labels(condition, split):
    return np.load(os.path.join(OUTPUT_ROOT, condition, split, "labels.npy"))


def _load_predictions(model_name, condition):
    val_rel, test_rel = MODEL_PRED_PATHS[model_name](condition)
    val_path = os.path.join(RESULTS_ROOT, val_rel)
    test_path = os.path.join(RESULTS_ROOT, test_rel)
    if not (os.path.exists(val_path) and os.path.exists(test_path)):
        return None, None
    return np.load(val_path), np.load(test_path)


def evaluate_one(model_name, condition):
    val_labels = _load_labels(condition, "val")
    test_labels = _load_labels(condition, "test")
    val_probs, test_probs = _load_predictions(model_name, condition)
    if val_probs is None:
        print(f"  SKIP {model_name}/{condition}: prediction files not found")
        return None

    # --- threshold-independent metrics, with bootstrap CIs (Section 3.6.2) ---
    auroc, auroc_lo, auroc_hi = bootstrap_ci(test_labels, test_probs, roc_auc_score)
    auprc, auprc_lo, auprc_hi = bootstrap_ci(test_labels, test_probs, average_precision_score)

    # --- threshold selection on VAL, applied to TEST (Section 3.6.2) ---
    threshold = best_f1_threshold(val_labels, val_probs)
    test_preds = (test_probs >= threshold).astype(int)
    f1, f1_lo, f1_hi = bootstrap_ci_f1(test_labels, test_probs, threshold)
    sensitivity, specificity = sensitivity_specificity(test_labels, test_preds)

    # --- calibration, before and after Platt scaling (Section 3.6.2) ---
    ece_before = expected_calibration_error(test_labels, test_probs)
    test_probs_calibrated = platt_scale(val_probs, val_labels, test_probs)
    ece_after = expected_calibration_error(test_labels, test_probs_calibrated)

    return {
        "model": model_name, "condition": condition,
        "auroc": auroc, "auroc_ci_lo": auroc_lo, "auroc_ci_hi": auroc_hi,
        "auprc": auprc, "auprc_ci_lo": auprc_lo, "auprc_ci_hi": auprc_hi,
        "f1": f1, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
        "threshold": threshold,
        "sensitivity": sensitivity, "specificity": specificity,
        "ece_before_platt": ece_before, "ece_after_platt": ece_after,
        "n_test": len(test_labels),
        "test_predictions": test_probs,  # kept in-memory only, for McNemar below; dropped before CSV export
        "test_threshold": threshold,
    }


def imputation_sensitivity(results_by_model):
    """Section 3.6.3: for each model, compare forward_fill vs
    linear_interp via (a) bootstrap CI overlap on AUROC and (b) McNemar's
    test on paired classification decisions -- computed TWO ways:

    1. "own-threshold": each condition's independently-selected
       F1-optimal threshold (Section 3.6.2's literal specification).
       This can conflate two distinct effects: the model's underlying
       risk SCORES shifting between conditions, and the THRESHOLD
       itself shifting (which alone can flip many borderline patients
       even if scores barely moved).
    2. "shared-threshold": both conditions' predictions binarised using
       forward_fill's threshold (forward_fill = the Harutyunyan et al.
       2019 standard condition, Section 3.3, used here as the reference).
       This isolates genuine score-level disagreement from
       threshold-selection artifacts, since the same cutoff is applied
       to both.

    Reporting both side by side lets the discussion chapter distinguish
    "this model's risk estimates genuinely shift with imputation" from
    "this model's optimal threshold happens to move, which alone
    explains most of the disagreement."
    """
    rows = []
    for model_name, by_condition in results_by_model.items():
        if "forward_fill" not in by_condition or "linear_interp" not in by_condition:
            continue
        ff = by_condition["forward_fill"]
        li = by_condition["linear_interp"]

        ci_overlap = not (ff["auroc_ci_hi"] < li["auroc_ci_lo"] or li["auroc_ci_hi"] < ff["auroc_ci_lo"])

        assert ff["n_test"] == li["n_test"], (
            f"{model_name}: test set sizes differ between conditions "
            f"({ff['n_test']} vs {li['n_test']}) -- cannot pair for McNemar"
        )

        # --- (1) own-threshold: each condition's own F1-optimal cutoff ---
        ff_preds_own = (ff["test_predictions"] >= ff["test_threshold"]).astype(int)
        li_preds_own = (li["test_predictions"] >= li["test_threshold"]).astype(int)
        stat_own, pvalue_own, ff_only_own, li_only_own = mcnemar_test(ff_preds_own, li_preds_own)

        # --- (2) shared-threshold: forward_fill's threshold applied to BOTH ---
        shared_threshold = ff["test_threshold"]
        ff_preds_shared = (ff["test_predictions"] >= shared_threshold).astype(int)
        li_preds_shared = (li["test_predictions"] >= shared_threshold).astype(int)
        stat_shared, pvalue_shared, ff_only_shared, li_only_shared = mcnemar_test(
            ff_preds_shared, li_preds_shared
        )

        rows.append({
            "model": model_name,
            "forward_fill_auroc": ff["auroc"],
            "linear_interp_auroc": li["auroc"],
            "auroc_delta": ff["auroc"] - li["auroc"],
            "ci_overlap": ci_overlap,
            "forward_fill_threshold": ff["test_threshold"],
            "linear_interp_threshold": li["test_threshold"],
            "threshold_delta": ff["test_threshold"] - li["test_threshold"],
            # own-threshold McNemar (Section 3.6.3 literal spec)
            "mcnemar_statistic_own_threshold": stat_own,
            "mcnemar_pvalue_own_threshold": pvalue_own,
            "n_ff_only_positive_own_threshold": ff_only_own,
            "n_li_only_positive_own_threshold": li_only_own,
            "significant_own_threshold": pvalue_own < 0.05,
            # shared-threshold McNemar (isolates score-level disagreement)
            "mcnemar_statistic_shared_threshold": stat_shared,
            "mcnemar_pvalue_shared_threshold": pvalue_shared,
            "n_ff_only_positive_shared_threshold": ff_only_shared,
            "n_li_only_positive_shared_threshold": li_only_shared,
            "significant_shared_threshold": pvalue_shared < 0.05,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = []
    results_by_model = {}
    for model_name in MODEL_PRED_PATHS:
        results_by_model[model_name] = {}
        for condition in IMPUTATION_CONDITIONS:
            print(f"Evaluating {model_name}/{condition}...")
            res = evaluate_one(model_name, condition)
            if res is not None:
                all_results.append(res)
                results_by_model[model_name][condition] = res

    if not all_results:
        print("\nNo prediction files found anywhere under results/ -- "
              "make sure the model scripts have been run first.")
        return

    # --- main results table (drop the in-memory-only prediction arrays) ---
    full_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "test_predictions"}
        for r in all_results
    ])
    full_path = os.path.join(OUT_DIR, "full_results.csv")
    full_df.to_csv(full_path, index=False)
    print(f"\nFull results written to {full_path}")

    # --- imputation sensitivity (Section 3.6.3, Research Question 3) ---
    sensitivity_df = imputation_sensitivity(results_by_model)
    sensitivity_path = os.path.join(OUT_DIR, "imputation_sensitivity.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"Imputation sensitivity results written to {sensitivity_path}")

    # --- human-readable summary ---
    summary_lines = ["# Evaluation Summary\n", "## Full results (test set)\n"]
    display_cols = ["model", "condition", "auroc", "auprc", "f1", "sensitivity",
                     "specificity", "ece_before_platt", "ece_after_platt"]
    summary_lines.append(full_df[display_cols].round(4).to_markdown(index=False))
    summary_lines.append("\n\n## Imputation sensitivity (McNemar test: own-threshold vs shared-threshold)\n")
    sensitivity_display_cols = [
        "model", "auroc_delta", "ci_overlap", "threshold_delta",
        "mcnemar_pvalue_own_threshold", "significant_own_threshold",
        "mcnemar_pvalue_shared_threshold", "significant_shared_threshold",
    ]
    summary_lines.append(sensitivity_df[sensitivity_display_cols].round(4).to_markdown(index=False))
    summary_path = os.path.join(OUT_DIR, "full_results_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Human-readable summary written to {summary_path}")

    print(f"\n{'=' * 70}\nFull results\n{'=' * 70}")
    print(full_df[display_cols].round(4).to_string(index=False))
    print(f"\n{'=' * 70}\nImputation sensitivity (own-threshold vs shared-threshold)\n{'=' * 70}")
    print(sensitivity_df[sensitivity_display_cols].round(4).to_string(index=False))
    print(f"\n(Full sensitivity detail, including raw discordant-pair counts "
          f"for both threshold variants, saved to {sensitivity_path})")


if __name__ == "__main__":
    main()