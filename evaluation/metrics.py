"""
evaluation/metrics.py

Core metric functions for Section 3.6:
  - AUROC, AUPRC (threshold-independent)
  - F1-maximising threshold selection on validation, applied to test
    (Section 3.6.2)
  - Sensitivity/specificity at that threshold
  - Expected Calibration Error (ECE), before and after Platt scaling
    (Section 3.6.2)
  - Bootstrap confidence intervals, n=1000 (Section 3.6.2)
  - McNemar's test on paired classification decisions, for the
    imputation-sensitivity analysis (Section 3.6.3)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix,
)
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

N_BOOTSTRAP = 1000
N_CALIBRATION_BINS = 10


def best_f1_threshold(y_true, y_prob):
    """Threshold on [0,1] that maximises F1 on the given (val) data.
    Searched over the set of observed probability values, which is
    where the optimal threshold for a finite sample must lie."""
    thresholds = np.unique(y_prob)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sensitivity, specificity


def expected_calibration_error(y_true, y_prob, n_bins=N_CALIBRATION_BINS):
    """Sample-weighted mean absolute difference between predicted
    probability and observed frequency, across n_bins equal-width bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return ece


def platt_scale(val_probs, val_labels, test_probs):
    """Fits a 1-D logistic regression (Platt scaling) mapping raw
    probabilities -> calibrated probabilities, fit on validation data
    only, applied to test."""
    calibrator = LogisticRegression()
    calibrator.fit(val_probs.reshape(-1, 1), val_labels)
    return calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=N_BOOTSTRAP, seed=42, **kwargs):
    """95% bootstrap confidence interval for a metric, resampling the
    test set with replacement. Skips degenerate resamples (single class
    only) for threshold-independent metrics that require both classes."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            stats.append(metric_fn(yt, yp, **kwargs))
        except Exception:
            continue
    stats = np.array(stats)
    point = metric_fn(y_true, y_prob, **kwargs)
    return point, np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def bootstrap_ci_f1(y_true, y_prob, threshold, n_boot=N_BOOTSTRAP, seed=42):
    """Bootstrap CI for F1 at a FIXED threshold (selected on val, not
    re-derived per resample -- Section 3.6.2)."""
    def f1_at_fixed_threshold(yt, yp):
        preds = (yp >= threshold).astype(int)
        return f1_score(yt, preds, zero_division=0)
    return bootstrap_ci(y_true, y_prob, f1_at_fixed_threshold, n_boot=n_boot, seed=seed)


def mcnemar_test(preds_a, preds_b):
    """McNemar's test on paired binary classification DECISIONS between
    two conditions (e.g. forward_fill vs linear_interp) for the same
    model on the same test patients, per Section 3.6.3. preds_a/preds_b
    must be paired (same order, same underlying patients).

    Returns (statistic, p_value, n_a_only, n_b_only) where n_a_only is
    the count of patients classified positive under condition A but not
    B, and vice versa -- the off-diagonal disagreement counts that
    actually drive the test.
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    assert len(preds_a) == len(preds_b), "predictions must be paired (same length)"

    both_pos = int(((preds_a == 1) & (preds_b == 1)).sum())
    a_only = int(((preds_a == 1) & (preds_b == 0)).sum())
    b_only = int(((preds_a == 0) & (preds_b == 1)).sum())
    both_neg = int(((preds_a == 0) & (preds_b == 0)).sum())

    table = [[both_pos, a_only], [b_only, both_neg]]
    # exact binomial test when discordant pairs are few (standard rule of thumb: <25)
    use_exact = (a_only + b_only) < 25
    result = sm_mcnemar(table, exact=use_exact, correction=not use_exact)
    return result.statistic, result.pvalue, a_only, b_only