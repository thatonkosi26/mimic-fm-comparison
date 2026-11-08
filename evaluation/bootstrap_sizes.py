"""
evaluation/bootstrap_sizes.py

Bootstrap confidence intervals for the Chronos size-comparison runs
(models/chronos_sizes.py), which evaluation/evaluate.py does not cover.

WHAT IT DOES
  1. For every size (mini/small/base) x head (exact_lr / exact_mlp) x
     imputation condition: point estimates and 95% bootstrap CIs for AUROC,
     AUPRC and F1, plus sensitivity and specificity. Same recipe as the main
     evaluation: 1,000 resamples of the test set with replacement, and the
     F1-optimal threshold chosen ON THE VALIDATION SET and held fixed.
  2. PAIRED bootstrap differences in AUROC between chosen pairs of models
     (for example Base vs Mini, MLP head vs LR head, frozen MLP vs the
     fine-tuned Chronos-Small, frozen MLP vs logistic regression). All models
     are scored on the SAME test patients, and every model uses the SAME
     bootstrap resamples, so a difference's CI is much tighter and fairer than
     comparing two separate CIs by eye. A difference is "significant" when its
     95% CI excludes 0.
  3. A CHECK that this script's threshold/F1/sensitivity/specificity code
     reproduces the numbers in results/evaluation/full_results.csv for the
     main models. If it does not, you are told, so you know whether the F1/
     sens/spec of the size runs are directly comparable with the main table.
     (AUROC and AUPRC do not depend on the threshold and are always comparable.)

Usage (from the project root; needs only numpy/pandas/scikit-learn):
    python evaluation\\bootstrap_sizes.py

Environment variables: N_BOOT (default 1000).

Output (results/evaluation/):
    bootstrap_sizes.csv           point estimates + 95% CIs, size runs
    bootstrap_sizes_paired.csv    paired AUROC differences, all pairs
    bootstrap_sizes_table.tex     LaTeX tables for the paper's appendix

Not included on purpose: TFT. Its predictions come from the pytorch-forecasting
dataloader, so their order should be checked against the label order before
being paired with other models. Add "tft" to REFERENCE_MODELS only after
checking that len and order match.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, RANDOM_SEED, PROJECT_ROOT

N_BOOT = int(os.environ.get("N_BOOT", 1000))
RES = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RES, "evaluation")

SIZES = ["mini", "small", "base"]
HEADS = ["exact_lr", "exact_mlp"]
SIZE_MODELS = [f"chronos_{s}_{h}" for s in SIZES for h in HEADS]
REFERENCE_MODELS = ["logistic_regression", "random_forest", "xgboost", "lstm",
                    "chronos_zeroshot", "chronos_finetuned"]

DISPLAY = {
    "logistic_regression": "Logistic regression", "random_forest": "Random Forest",
    "xgboost": "XGBoost", "lstm": "LSTM",
    "chronos_zeroshot": "Chronos-Small zero-shot (sampled)",
    "chronos_finetuned": "Chronos-Small fine-tuned",
}
for _s in SIZES:
    DISPLAY[f"chronos_{_s}_exact_lr"] = f"Chronos-{_s.capitalize()} frozen, LR head"
    DISPLAY[f"chronos_{_s}_exact_mlp"] = f"Chronos-{_s.capitalize()} frozen, MLP head"

COND_DISPLAY = {"forward_fill": "Forward-fill", "linear_interp": "Linear interp."}


# ---------------------------------------------------------------- loading
def load_labels(cond, split):
    return np.load(os.path.join(OUTPUT_ROOT, cond, split, "labels.npy")).astype(int)


def pred_path(cond, name, split):
    parts = name.split("_", 2)
    if name.startswith("chronos_") and len(parts) == 3 and parts[1] in SIZES:
        _, size, head = parts
        return os.path.join(RES, "chronos_sizes", size, cond, f"{head}_{split}_predictions.npy")
    if name in ("logistic_regression", "random_forest", "xgboost"):
        return os.path.join(RES, "baselines", cond, f"{name}_{split}_predictions.npy")
    if name == "lstm":
        return os.path.join(RES, "lstm", cond, f"{split}_predictions.npy")
    if name == "chronos_zeroshot":
        return os.path.join(RES, "chronos", cond, f"zeroshot_{split}_predictions.npy")
    if name == "chronos_finetuned":
        return os.path.join(RES, "chronos", cond, f"finetuned_{split}_predictions.npy")
    raise ValueError(name)


def load_pred(cond, name, split, n_expected):
    p = np.load(pred_path(cond, name, split)).astype(float).reshape(-1)
    if len(p) != n_expected:
        raise ValueError(f"{name}/{cond}/{split}: {len(p)} predictions but {n_expected} labels")
    return p


# ---------------------------------------------------------------- metrics
def f1_threshold(y, p):
    """Threshold maximising F1 on (y, p) -- used on the VALIDATION set only."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def counts(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def f1_sens_spec(y, p, thr):
    tp, fp, fn, tn = counts(y, (p >= thr).astype(int))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return f1, sens, spec


def bootstrap_arrays(y, p, thr, idx):
    """AUROC, AUPRC, F1 for each bootstrap resample (rows of idx)."""
    B = len(idx)
    au, ap, f1 = (np.full(B, np.nan) for _ in range(3))
    for b in range(B):
        i = idx[b]
        yb, pb = y[i], p[i]
        if yb.min() == yb.max():
            continue
        au[b] = roc_auc_score(yb, pb)
        ap[b] = average_precision_score(yb, pb)
        tp, fp, fn, _ = counts(yb, (pb >= thr).astype(int))
        f1[b] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return au, ap, f1


def ci(arr):
    return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


# ---------------------------------------------------------------- main
def run_condition(cond):
    y_val, y_test = load_labels(cond, "val"), load_labels(cond, "test")
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.integers(0, len(y_test), size=(N_BOOT, len(y_test)))  # shared resamples (paired)

    store, rows = {}, []
    for name in SIZE_MODELS + REFERENCE_MODELS:
        try:
            p_val = load_pred(cond, name, "val", len(y_val))
            p_test = load_pred(cond, name, "test", len(y_test))
        except FileNotFoundError as e:
            print(f"  [skip] {name}/{cond}: {e.filename} not found")
            continue
        thr = f1_threshold(y_val, p_val)
        au_b, ap_b, f1_b = bootstrap_arrays(y_test, p_test, thr, idx)
        store[name] = {"au": au_b, "thr": thr, "p_test": p_test}
        f1, sens, spec = f1_sens_spec(y_test, p_test, thr)
        au, ap = roc_auc_score(y_test, p_test), average_precision_score(y_test, p_test)
        rows.append({
            "model": name, "condition": cond, "threshold": thr,
            "auroc": au, "auroc_ci_lo": ci(au_b)[0], "auroc_ci_hi": ci(au_b)[1],
            "auprc": ap, "auprc_ci_lo": ci(ap_b)[0], "auprc_ci_hi": ci(ap_b)[1],
            "f1": f1, "f1_ci_lo": ci(f1_b)[0], "f1_ci_hi": ci(f1_b)[1],
            "sensitivity": sens, "specificity": spec, "n_test": len(y_test),
        })
    return rows, store


def paired_pairs():
    pairs = []
    for h in HEADS:                                   # size effect within a head
        pairs += [(f"chronos_base_{h}", f"chronos_mini_{h}"),
                  (f"chronos_base_{h}", f"chronos_small_{h}"),
                  (f"chronos_small_{h}", f"chronos_mini_{h}")]
    for s in SIZES:                                   # head effect within a size
        pairs.append((f"chronos_{s}_exact_mlp", f"chronos_{s}_exact_lr"))
    for s in SIZES:                                   # frozen MLP vs main models
        for ref in ["chronos_finetuned", "chronos_zeroshot", "logistic_regression",
                    "random_forest", "xgboost", "lstm"]:
            pairs.append((f"chronos_{s}_exact_mlp", ref))
    for s in SIZES:                                   # frozen LR vs sampled zero-shot / fine-tuned
        pairs += [(f"chronos_{s}_exact_lr", "chronos_zeroshot"),
                  (f"chronos_{s}_exact_lr", "chronos_finetuned")]
    return pairs


def paired_rows(cond, store):
    out = []
    for a, b in paired_pairs():
        if a not in store or b not in store:
            continue
        d = store[a]["au"] - store[b]["au"]
        diff_point = (roc_auc_score(load_labels(cond, "test"), store[a]["p_test"])
                      - roc_auc_score(load_labels(cond, "test"), store[b]["p_test"]))
        lo, hi = ci(d)
        p = min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))
        out.append({"condition": cond, "model_a": a, "model_b": b,
                    "auroc_diff_a_minus_b": diff_point, "ci_lo": lo, "ci_hi": hi,
                    "bootstrap_p": float(p), "significant_95": bool(lo > 0 or hi < 0)})
    return out


def check_against_main(cond, store):
    """Does this script reproduce evaluation/evaluate.py's numbers for the main models?"""
    path = os.path.join(OUT_DIR, "full_results.csv")
    if not os.path.exists(path):
        return None
    ref = pd.read_csv(path)
    ref = ref[ref["condition"] == cond]
    worst = {"threshold": 0.0, "f1": 0.0, "sensitivity": 0.0, "specificity": 0.0, "auroc": 0.0}
    y_test = load_labels(cond, "test")
    n = 0
    for name in REFERENCE_MODELS:
        r = ref[ref["model"] == name]
        if r.empty or name not in store:
            continue
        r = r.iloc[0]
        thr = store[name]["thr"]
        f1, sens, spec = f1_sens_spec(y_test, store[name]["p_test"], thr)
        worst["threshold"] = max(worst["threshold"], abs(thr - r["threshold"]))
        worst["f1"] = max(worst["f1"], abs(f1 - r["f1"]))
        worst["sensitivity"] = max(worst["sensitivity"], abs(sens - r["sensitivity"]))
        worst["specificity"] = max(worst["specificity"], abs(spec - r["specificity"]))
        worst["auroc"] = max(worst["auroc"], abs(roc_auc_score(y_test, store[name]["p_test"]) - r["auroc"]))
        n += 1
    return worst, n


def fmt(x):
    return f"{x:.4f}"


def latex_tables(df, paired):
    t1 = [r"\begin{table*}[!t]", r"\centering",
          r"\caption{Chronos size comparison (frozen weights, exact features): test-set point estimates with 95\% bootstrap confidence intervals (1,000 resamples).}",
          r"\label{tab:ci-sizes}", r"\footnotesize", r"\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{llllll}", r"\toprule",
          r"\textbf{Model} & \textbf{Condition} & \textbf{AUROC} & \textbf{AUPRC} & \textbf{F1} & \textbf{Sens. / Spec.} \\",
          r"\midrule"]
    sz = df[df["model"].isin(SIZE_MODELS)]
    last = None
    for _, r in sz.sort_values(["model", "condition"], key=lambda c: c.map(
            {m: i for i, m in enumerate(SIZE_MODELS)}) if c.name == "model" else c).iterrows():
        if last is not None and last != r["model"]:
            t1.append(r"\midrule")
        last = r["model"]
        t1.append(f"{DISPLAY[r['model']]} & {COND_DISPLAY[r['condition']]} & "
                  f"{fmt(r['auroc'])} [{fmt(r['auroc_ci_lo'])}, {fmt(r['auroc_ci_hi'])}] & "
                  f"{fmt(r['auprc'])} [{fmt(r['auprc_ci_lo'])}, {fmt(r['auprc_ci_hi'])}] & "
                  f"{fmt(r['f1'])} [{fmt(r['f1_ci_lo'])}, {fmt(r['f1_ci_hi'])}] & "
                  f"{r['sensitivity']:.3f} / {r['specificity']:.3f} \\\\")
    t1 += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]

    pick = [("chronos_base_exact_lr", "chronos_mini_exact_lr"),
            ("chronos_base_exact_mlp", "chronos_mini_exact_mlp"),
            ("chronos_small_exact_mlp", "chronos_small_exact_lr"),
            ("chronos_small_exact_lr", "chronos_zeroshot"),
            ("chronos_small_exact_mlp", "chronos_finetuned"),
            ("chronos_base_exact_mlp", "logistic_regression"),
            ("chronos_base_exact_mlp", "xgboost")]
    t2 = [r"\begin{table*}[!t]", r"\centering",
          r"\caption{Paired bootstrap AUROC differences (A minus B) on the same test patients. A difference is significant when its 95\% CI excludes 0.}",
          r"\label{tab:paired}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
          r"\begin{tabular}{p{4.6cm}p{4.6cm}lcc}", r"\toprule",
          r"\textbf{A} & \textbf{B} & \textbf{Cond.} & \textbf{Diff. [95\% CI]} & \textbf{Sig.} \\", r"\midrule"]
    for a, b in pick:
        for cond in IMPUTATION_CONDITIONS:
            r = paired[(paired["model_a"] == a) & (paired["model_b"] == b) & (paired["condition"] == cond)]
            if r.empty:
                continue
            r = r.iloc[0]
            short = {"forward_fill": "FF", "linear_interp": "LI"}[cond]
            t2.append(f"{DISPLAY[a]} & {DISPLAY[b]} & {short} & "
                      f"{r['auroc_diff_a_minus_b']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] & "
                      f"{'yes' if r['significant_95'] else 'no'} \\\\")
    t2 += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(t1 + t2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows, all_paired = [], []
    for cond in IMPUTATION_CONDITIONS:
        print(f"\n=== {cond} ===")
        rows, store = run_condition(cond)
        all_rows += rows
        all_paired += paired_rows(cond, store)
        chk = check_against_main(cond, store)
        if chk is None:
            print("  (check skipped: results/evaluation/full_results.csv not found)")
        else:
            worst, n = chk
            print(f"  Check vs full_results.csv on {n} main models, largest differences: "
                  + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))
            if worst["threshold"] > 1e-6 or worst["f1"] > 1e-6:
                print("  !! THRESHOLD/F1 DO NOT MATCH evaluate.py exactly. AUROC/AUPRC and all CIs for them "
                      "are unaffected, but F1/sens/spec of the size runs may differ slightly in definition "
                      "from the main table. Send me the difference above and I will align the method.")
            else:
                print("  OK: threshold, F1, sensitivity and specificity reproduce the main table.")

    df = pd.DataFrame(all_rows)
    paired = pd.DataFrame(all_paired)
    df.to_csv(os.path.join(OUT_DIR, "bootstrap_sizes.csv"), index=False)
    paired.to_csv(os.path.join(OUT_DIR, "bootstrap_sizes_paired.csv"), index=False)
    with open(os.path.join(OUT_DIR, "bootstrap_sizes_table.tex"), "w") as f:
        f.write(latex_tables(df, paired))

    pd.set_option("display.width", 200)
    print("\nSize runs (AUROC with 95% CI):")
    for _, r in df[df["model"].isin(SIZE_MODELS)].iterrows():
        print(f"  {r['model']:<26} {r['condition']:<14} {r['auroc']:.4f} [{r['auroc_ci_lo']:.4f}, {r['auroc_ci_hi']:.4f}]")
    print("\nPaired AUROC differences (A - B), selected:")
    show = paired[paired["model_b"].isin(["chronos_mini_exact_lr", "chronos_mini_exact_mlp", "chronos_finetuned",
                                          "logistic_regression", "chronos_small_exact_lr", "chronos_zeroshot"])]
    for _, r in show.iterrows():
        print(f"  {r['condition']:<13} {r['model_a']:<24} - {r['model_b']:<22} "
              f"{r['auroc_diff_a_minus_b']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
              f"{'SIGNIFICANT' if r['significant_95'] else 'not significant'}")
    print(f"\nWrote files to {OUT_DIR}")


if __name__ == "__main__":
    main()