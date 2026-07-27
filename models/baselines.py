"""
models/baselines.py

Implements Section 3.5.1: logistic regression, Random Forest, and XGBoost,
trained on the 102-dim static feature vectors produced by build_dataset.py.
Applies the uniform class-imbalance handling from Section 3.4
(class_weight='balanced' / scale_pos_weight), and the CV search spaces
exactly as specified in the proposal.

Trains and evaluates each model under BOTH imputation conditions
(forward_fill, linear_interp), since Research Question 3 requires a
within-model, cross-condition comparison later (see evaluation/).

Usage:
    python models/baselines.py

Output:
    results/baselines/<condition>/<model_name>_model.joblib
    results/baselines/<condition>/<model_name>_val_predictions.npy
    results/baselines/<condition>/<model_name>_test_predictions.npy
    results/baselines/summary.csv   (val AUROC per model x condition, for a quick look)
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from joblib import dump
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, RANDOM_SEED, PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "baselines")
CV_FOLDS = 5
N_RANDOM_ITER = 50

# IMPORTANT (Windows): each joblib worker process re-imports numpy/scipy/
# sklearn from scratch (no fork on Windows, only spawn), which costs real
# RAM per worker. n_jobs=-1 (one worker per core) caused an out-of-memory
# crash even on a small grid search. Override via the N_JOBS env var if you
# want to tune this for your machine, e.g. `set N_JOBS=2` before running.
N_JOBS = int(os.environ.get("N_JOBS", min(4, os.cpu_count() or 4)))


def _load_split(condition, split):
    base = os.path.join(OUTPUT_ROOT, condition, split)
    X = np.load(os.path.join(base, "static_feats.npy"))
    y = np.load(os.path.join(base, "labels.npy"))
    return X, y


def _neg_pos_ratio(y):
    """Ratio of negative to positive training examples, used for
    scale_pos_weight (XGBoost) per Section 3.4."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return n_neg / max(n_pos, 1)


def train_logistic_regression(X_train, y_train):
    """Section 3.5.1: L2-regularised, class_weight='balanced', C searched
    over {0.001, 0.01, 0.1, 1, 10} via 5-fold CV.

    The 102 static features are on very different scales (GCS scores
    ~3-15, heart rate ~60-150, observation counts 0-48), which makes
    lbfgs converge slowly/poorly for logistic regression specifically
    (tree-based models are scale-invariant, so RF/XGBoost don't need
    this). Scaling is fit *inside* the CV pipeline so it's refit on each
    training fold -- no leakage from validation folds into the scaler.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000,
            random_state=RANDOM_SEED, solver="lbfgs",
        )),
    ])
    param_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10]}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=N_JOBS)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_random_forest(X_train, y_train):
    """Section 3.5.1: 500 trees, class_weight='balanced', max_depth and
    min_samples_leaf tuned via 50-iteration random search, 5-fold CV."""
    param_dist = {
        "max_depth": randint(4, 30),
        "min_samples_leaf": randint(1, 20),
    }
    base = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_RANDOM_ITER, scoring="roc_auc",
        cv=cv, n_jobs=N_JOBS, random_state=RANDOM_SEED,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_xgboost(X_train, y_train):
    """Section 3.5.1: scale_pos_weight = neg/pos training ratio; learning
    rate, max_depth, n_estimators, subsample, L1/L2 coefs tuned via
    50-iteration random search, 5-fold CV."""
    spw = _neg_pos_ratio(y_train)
    param_dist = {
        "learning_rate": uniform(0.01, 0.29),       # ~0.01-0.30
        "max_depth": randint(3, 10),
        "n_estimators": randint(100, 600),
        "subsample": uniform(0.6, 0.4),              # ~0.6-1.0
        "reg_alpha": uniform(0, 1),                  # L1
        "reg_lambda": uniform(0.5, 2.5),              # L2, ~0.5-3.0
    }
    base = XGBClassifier(
        scale_pos_weight=spw, eval_metric="auc",
        random_state=RANDOM_SEED, n_jobs=1, tree_method="hist",
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=N_RANDOM_ITER, scoring="roc_auc",
        cv=cv, n_jobs=N_JOBS, random_state=RANDOM_SEED,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


TRAINERS = {
    "logistic_regression": train_logistic_regression,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}


def run_condition(condition):
    print(f"\n{'=' * 60}\nCondition: {condition}\n{'=' * 60}")
    X_train, y_train = _load_split(condition, "train")
    X_val, y_val = _load_split(condition, "val")
    X_test, y_test = _load_split(condition, "test")

    out_dir = os.path.join(RESULTS_DIR, condition)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for name, trainer_fn in TRAINERS.items():
        print(f"\n--- Training {name} ---")
        model, best_params = trainer_fn(X_train, y_train)
        print(f"  best params: {best_params}")

        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        val_auroc = roc_auc_score(y_val, val_probs)
        test_auroc = roc_auc_score(y_test, test_probs)
        print(f"  val AUROC: {val_auroc:.4f}   test AUROC: {test_auroc:.4f}")

        dump(model, os.path.join(out_dir, f"{name}_model.joblib"))
        np.save(os.path.join(out_dir, f"{name}_val_predictions.npy"), val_probs)
        np.save(os.path.join(out_dir, f"{name}_test_predictions.npy"), test_probs)
        with open(os.path.join(out_dir, f"{name}_best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2, default=str)

        rows.append({
            "condition": condition, "model": name,
            "val_auroc": val_auroc, "test_auroc": test_auroc,
            "best_params": json.dumps(best_params, default=str),
        })
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    for condition in IMPUTATION_CONDITIONS:
        all_rows.extend(run_condition(condition))

    summary = pd.DataFrame(all_rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 60}\nSummary written to {summary_path}\n{'=' * 60}")
    print(summary[["condition", "model", "val_auroc", "test_auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()