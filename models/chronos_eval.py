"""
models/chronos_eval.py

Implements Section 3.5.3's ZERO-SHOT Chronos configuration:
  - Each of the 17 variable channels passed independently to a frozen
    pretrained Chronos model as a 48-step context window.
  - For each channel, extract mean and variance of the predictive
    distribution over the next step -> 34-dim feature vector per episode
    (17 channels x 2 stats).
  - Logistic regression (class_weight='balanced') trained on the
    34-dim features. Note: the proposal does NOT specify a C-search for
    this classifier (unlike the tuned LR baseline in Section 3.5.1), so
    this uses sklearn's default C=1.0 -- a deliberate, faithful reading
    of the proposal text, not an oversight.

MODEL SIZE: uses Chronos-Small (46M params) rather than Chronos-Large
(710M), a resource-driven deviation from the proposal's default choice,
explicitly permitted by Section 3.5.3 ("If the Large model exceeds
available GPU memory, I will fall back to the Small or Mini variant,
and the model size used will be reported explicitly"). On this CPU-only
machine, benchmarking showed Large would be impractical; Small was
chosen as the best available balance of capability vs. runtime
(~17 hours estimated for the full zero-shot pass across both imputation
conditions, per scripts/benchmark_chronos.py).

CHECKPOINTING: given the long runtime, results are cached PER CHANNEL
PER SPLIT PER CONDITION as they complete. If interrupted, re-running
this script picks up from the next incomplete channel rather than
starting over -- don't delete results/chronos/cache/ between runs.

Usage:
    python models/chronos_eval.py                 # full run
    set CHRONOS_QUICK_MODE=1 & python models/chronos_eval.py   # test on a small subset first (STRONGLY recommended before the full run)

Output:
    results/chronos/cache/<condition>_<split>_channel<NN>.npz   (checkpoints)
    results/chronos/<condition>/zeroshot_features_<split>.npy  (N, 34)
    results/chronos/<condition>/zeroshot_model.joblib
    results/chronos/<condition>/zeroshot_val_predictions.npy
    results/chronos/<condition>/zeroshot_test_predictions.npy
    results/chronos/zeroshot_summary.csv
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, RANDOM_SEED, PROJECT_ROOT, VARIABLES

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "chronos")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")

MODEL_NAME = os.environ.get("CHRONOS_MODEL", "amazon/chronos-t5-small")
BATCH_SIZE = int(os.environ.get("CHRONOS_BATCH_SIZE", 32))
NUM_SAMPLES = int(os.environ.get("CHRONOS_NUM_SAMPLES", 20))  # explicit for reproducibility (Section 3.7)
QUICK_MODE = os.environ.get("CHRONOS_QUICK_MODE", "0") == "1"
QUICK_MODE_N = int(os.environ.get("CHRONOS_QUICK_N", 50))

SEQ_LEN = 48
N_CHANNELS = len(VARIABLES)


def _load_split(condition, split):
    base = os.path.join(OUTPUT_ROOT, condition, split)
    X = np.load(os.path.join(base, "sequences.npy")).astype(np.float32)
    y = np.load(os.path.join(base, "labels.npy")).astype(int)
    if QUICK_MODE:
        X, y = X[:QUICK_MODE_N], y[:QUICK_MODE_N]
    return X, y


def _channel_cache_path(condition, split, channel_idx):
    tag = "_quick" if QUICK_MODE else ""
    return os.path.join(CACHE_DIR, f"{condition}_{split}_channel{channel_idx:02d}{tag}.npz")


def _compute_channel_stats(pipeline, X_channel, condition, split, channel_idx):
    """Returns (means, variances) each shape (N,) for one channel across
    all episodes in this split, using per-channel checkpointing so an
    interrupted run can resume."""
    cache_path = _channel_cache_path(condition, split, channel_idx)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["means"], cached["variances"]

    N = X_channel.shape[0]
    means = np.zeros(N, dtype=np.float32)
    variances = np.zeros(N, dtype=np.float32)

    context_tensor = torch.from_numpy(X_channel)  # (N, 48)

    t0 = time.time()
    for i in range(0, N, BATCH_SIZE):
        batch = context_tensor[i:i + BATCH_SIZE]
        samples = pipeline.predict(
            inputs=batch, prediction_length=1, num_samples=NUM_SAMPLES,
        )  # (batch, num_samples, 1)
        means[i:i + BATCH_SIZE] = samples.mean(dim=1).squeeze(-1).numpy()
        variances[i:i + BATCH_SIZE] = samples.var(dim=1).squeeze(-1).numpy()

    elapsed = time.time() - t0
    print(f"    channel {channel_idx:2d} ({VARIABLES[channel_idx]}): "
          f"{N} episodes in {elapsed:.1f}s ({elapsed / max(N, 1) * 1000:.1f}ms/episode)")

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, means=means, variances=variances)
    return means, variances


def extract_features(pipeline, condition, split):
    """Returns (features (N, 34), labels (N,))."""
    X, y = _load_split(condition, split)
    print(f"  extracting zero-shot features: {condition}/{split} ({len(y)} episodes)")

    all_means, all_vars = [], []
    for c in range(N_CHANNELS):
        means, variances = _compute_channel_stats(pipeline, X[:, :, c], condition, split, c)
        all_means.append(means)
        all_vars.append(variances)

    # 34-dim: 17 means then 17 variances (column order is arbitrary for a
    # linear classifier, but documented here for clarity/reproducibility)
    features = np.concatenate(
        [np.stack(all_means, axis=1), np.stack(all_vars, axis=1)], axis=1
    )
    return features.astype(np.float32), y


def run_condition(pipeline, condition):
    print(f"\n{'=' * 60}\nCondition: {condition}\n{'=' * 60}")
    out_dir = os.path.join(RESULTS_DIR, condition)
    os.makedirs(out_dir, exist_ok=True)

    feats_train, y_train = extract_features(pipeline, condition, "train")
    feats_val, y_val = extract_features(pipeline, condition, "val")
    feats_test, y_test = extract_features(pipeline, condition, "test")

    for name, feats in [("train", feats_train), ("val", feats_val), ("test", feats_test)]:
        np.save(os.path.join(out_dir, f"zeroshot_features_{name}.npy"), feats)

    # Section 3.5.3: logistic regression, class_weight='balanced', no
    # C-search specified for this classifier (unlike Section 3.5.1's
    # tuned baseline) -- default C=1.0 used deliberately.
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(feats_train, y_train)

    val_probs = clf.predict_proba(feats_val)[:, 1]
    test_probs = clf.predict_proba(feats_test)[:, 1]
    val_auroc = roc_auc_score(y_val, val_probs)
    test_auroc = roc_auc_score(y_test, test_probs)
    print(f"  zero-shot val AUROC: {val_auroc:.4f}   test AUROC: {test_auroc:.4f}")

    dump(clf, os.path.join(out_dir, "zeroshot_model.joblib"))
    np.save(os.path.join(out_dir, "zeroshot_val_predictions.npy"), val_probs)
    np.save(os.path.join(out_dir, "zeroshot_test_predictions.npy"), test_probs)

    return {
        "condition": condition, "model": "chronos_zeroshot",
        "chronos_model": MODEL_NAME,
        "val_auroc": val_auroc, "test_auroc": test_auroc,
    }


def main():
    torch.manual_seed(RANDOM_SEED)

    print(f"Loading {MODEL_NAME}...")
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME, device_map="cpu", torch_dtype=torch.float32,
    )
    n_params = sum(p.numel() for p in pipeline.model.parameters())
    print(f"Loaded. Params: {n_params / 1e6:.1f}M")

    if QUICK_MODE:
        print(f"*** QUICK MODE: using only the first {QUICK_MODE_N} episodes per "
              f"split. This is for testing correctness only -- results are NOT "
              f"valid for the dissertation. Unset CHRONOS_QUICK_MODE for the real run. ***")
    else:
        print("NOTE: this is the longest-running step in the pipeline. Progress "
              "is checkpointed per channel per split -- if interrupted, just "
              "re-run this script and it will skip already-completed channels "
              "rather than starting over.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = [run_condition(pipeline, cond) for cond in IMPUTATION_CONDITIONS]

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, "zeroshot_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 60}\nSummary written to {summary_path}\n{'=' * 60}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()