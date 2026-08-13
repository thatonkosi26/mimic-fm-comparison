"""
evaluation/generate_figures.py

Generates the figures for the Results chapter, reusing the exact same
prediction-loading logic as evaluate.py so the figures are guaranteed
consistent with the numbers already in results/evaluation/full_results.csv
(run evaluate.py first).

Usage:
    python evaluation/generate_figures.py

Output (300 DPI PNG, suitable for Word; also saves a PDF copy of each
for LaTeX users):
    results/figures/roc_curves_forward_fill.png(.pdf)
    results/figures/roc_curves_linear_interp.png(.pdf)
    results/figures/calibration_forward_fill.png(.pdf)
    results/figures/calibration_linear_interp.png(.pdf)
    results/figures/model_comparison_auroc.png(.pdf)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMPUTATION_CONDITIONS, PROJECT_ROOT

# Reuse the exact loading logic already verified in evaluate.py, rather
# than duplicating (and risking drift from) the file-path mapping.
from evaluate import load_predictions, load_labels
from metrics import platt_scale

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
FIG_DIR = os.path.join(RESULTS_ROOT, "figures")

# Consistent colour per model across every figure, and readable display names.
MODEL_DISPLAY = {
    "logistic_regression": ("Logistic Regression", "#4C72B0"),
    "random_forest": ("Random Forest", "#DD8452"),
    "xgboost": ("XGBoost", "#55A868"),
    "lstm": ("LSTM", "#C44E52"),
    "tft": ("TFT", "#8172B2"),
    "chronos_zeroshot": ("Chronos (zero-shot)", "#937860"),
    "chronos_finetuned": ("Chronos (fine-tuned)", "#DA8BC3"),
}

CONDITION_DISPLAY = {"forward_fill": "Forward-Fill Imputation", "linear_interp": "Linear Interpolation"}


def plot_roc_curves(condition):
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, (label, color) in MODEL_DISPLAY.items():
        _, test_probs = load_predictions(model_name, condition)
        if test_probs is None:
            continue
        test_labels = load_labels(condition, "test")
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        auroc = roc_auc_score(test_labels, test_probs)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc:.3f})", color=color, linewidth=1.8)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {CONDITION_DISPLAY[condition]}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    _save(fig, f"roc_curves_{condition}")


def plot_calibration(condition):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, calibrated in zip(axes, [False, True]):
        for model_name, (label, color) in MODEL_DISPLAY.items():
            val_probs, test_probs = load_predictions(model_name, condition)
            if val_probs is None:
                continue
            val_labels = load_labels(condition, "val")
            test_labels = load_labels(condition, "test")

            probs_to_plot = test_probs
            if calibrated:
                probs_to_plot = platt_scale(val_probs, val_labels, test_probs)

            frac_pos, mean_pred = calibration_curve(test_labels, probs_to_plot, n_bins=10, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker="o", markersize=3, label=label, color=color, linewidth=1.5)

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title("After Platt scaling" if calibrated else "Raw model output")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Calibration — {CONDITION_DISPLAY[condition]}", fontsize=13)
    fig.tight_layout()
    _save(fig, f"calibration_{condition}")


def plot_model_comparison():
    results_path = os.path.join(RESULTS_ROOT, "evaluation", "full_results.csv")
    if not os.path.exists(results_path):
        print(f"  {results_path} not found -- run evaluate.py first. Skipping comparison chart.")
        return
    df = pd.read_csv(results_path)

    fig, ax = plt.subplots(figsize=(11, 6))
    model_order = list(MODEL_DISPLAY.keys())
    x = np.arange(len(model_order))
    width = 0.35

    for i, condition in enumerate(IMPUTATION_CONDITIONS):
        sub = df[df["condition"] == condition].set_index("model").reindex(model_order)
        aurocs = sub["auroc"].values
        err_lo = aurocs - sub["auroc_ci_lo"].values
        err_hi = sub["auroc_ci_hi"].values - aurocs
        offset = (i - 0.5) * width
        ax.bar(
            x + offset, aurocs, width, yerr=[err_lo, err_hi], capsize=3,
            label=CONDITION_DISPLAY.get(condition, condition),
            color=["#4C72B0", "#DD8452"][i % 2],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m][0] for m in model_order], rotation=30, ha="right")
    ax.set_ylabel("Test AUROC (with 95% bootstrap CI)")
    ax.set_title("Model Comparison — Test AUROC by Imputation Condition")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.legend()
    # Size the y-axis to the actual data (with padding) rather than a fixed
    # guess -- a hardcoded range can silently clip bars/error bars if any
    # model's AUROC (or its CI) falls outside an assumed range.
    y_min = min(0.5, df["auroc_ci_lo"].min() - 0.02)
    y_max = df["auroc_ci_hi"].max() + 0.03
    ax.set_ylim([y_min, y_max])
    fig.tight_layout()
    _save(fig, "model_comparison_auroc")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png_path}")


def main():
    print("Generating ROC curves...")
    for condition in IMPUTATION_CONDITIONS:
        plot_roc_curves(condition)

    print("Generating calibration plots...")
    for condition in IMPUTATION_CONDITIONS:
        plot_calibration(condition)

    print("Generating model comparison chart...")
    plot_model_comparison()

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
