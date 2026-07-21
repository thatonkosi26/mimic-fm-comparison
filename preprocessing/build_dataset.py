"""
build_dataset.py

Orchestrates Section 3.3-3.5.1 end to end:
  1. Read the benchmark repo's train/val/test listfiles (patient -> label).
  2. Compute training-set means (for imputation fallback) ONCE per condition.
  3. Compute training-set z-score normalisation stats (mean/std) on the
     IMPUTED training grids, ONCE per condition -- never touching val/test.
  4. For every split and every condition, produce:
       - sequences.npy   : (N, 48, 17) normalised array for LSTM / TFT / Chronos
       - static_feats.npy: (N, 102) feature vector for LR / RF / XGBoost
       - labels.npy      : (N,) binary mortality labels
       - subject_ids.npy : (N,) for traceability / error analysis
  Output layout:
    OUTPUT_ROOT/<condition>/<split>/{sequences,static_feats,labels,subject_ids}.npy
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_DIR, TEST_DIR, TRAIN_LISTFILE, VAL_LISTFILE, TEST_LISTFILE,
    OUTPUT_ROOT, VARIABLES, IMPUTATION_CONDITIONS, RANDOM_SEED,
)
from imputation import (
    load_episode, condition_a_forward_fill, condition_b_linear_interp,
    compute_training_means,
)
from feature_extraction import extract_static_features

np.random.seed(RANDOM_SEED)

CONDITION_FUNCS = {
    "forward_fill": condition_a_forward_fill,
    "linear_interp": condition_b_linear_interp,
}


def _read_listfile(path: str, data_dir: str):
    """listfile.csv format from the benchmark repo: stay,y_true (+ header).
    Returns list of (full_csv_path, label, subject_identifier)."""
    lf = pd.read_csv(path)
    out = []
    for _, row in lf.iterrows():
        fname = row["stay"] if "stay" in lf.columns else row.iloc[0]
        label = int(row["y_true"] if "y_true" in lf.columns else row.iloc[1])
        out.append((os.path.join(data_dir, fname), label, fname))
    return out


def _normalise(values_df, means, stds):
    out = values_df.copy()
    for v in VARIABLES:
        s = stds[v] if stds[v] > 1e-8 else 1.0
        out[v] = (out[v] - means[v]) / s
    return out


def _fit_normalisation_stats(episode_records, impute_fn, train_means):
    """Pass 1 over training data (post-imputation) to get mean/std per
    variable for z-score normalisation, per Section 3.3."""
    all_vals = {v: [] for v in VARIABLES}
    for path, _, _ in tqdm(episode_records, desc="fitting norm stats"):
        df = load_episode(path)
        values_df, _ = impute_fn(df, train_means)
        for v in VARIABLES:
            all_vals[v].append(values_df[v].to_numpy())
    means, stds = {}, {}
    for v in VARIABLES:
        arr = np.concatenate(all_vals[v])
        means[v] = float(np.mean(arr))
        stds[v] = float(np.std(arr))
    return means, stds


def _process_split(split_name, episode_records, impute_fn, train_means, norm_means, norm_stds, out_dir):
    seqs, feats, labels, ids = [], [], [], []
    for path, label, sid in tqdm(episode_records, desc=f"processing {split_name}"):
        df = load_episode(path)
        values_df, mask_df = impute_fn(df, train_means)
        static_feat = extract_static_features(values_df, mask_df)
        norm_values = _normalise(values_df, norm_means, norm_stds)
        seqs.append(norm_values[VARIABLES].to_numpy(dtype=float))
        feats.append(static_feat)
        labels.append(label)
        ids.append(sid)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "sequences.npy"), np.stack(seqs))       # (N,48,17)
    np.save(os.path.join(out_dir, "static_feats.npy"), np.stack(feats))  # (N,102)
    np.save(os.path.join(out_dir, "labels.npy"), np.array(labels))
    np.save(os.path.join(out_dir, "subject_ids.npy"), np.array(ids))
    print(f"  {split_name}: {len(labels)} episodes, "
          f"mortality rate = {np.mean(labels):.3f}")


def main():
    train_records = _read_listfile(TRAIN_LISTFILE, TRAIN_DIR)
    val_records = _read_listfile(VAL_LISTFILE, TRAIN_DIR)
    test_records = _read_listfile(TEST_LISTFILE, TEST_DIR)

    train_paths = [p for p, _, _ in train_records]

    for cond_name, impute_fn in CONDITION_FUNCS.items():
        print(f"\n=== Condition: {cond_name} ===")
        train_means = compute_training_means(train_paths)
        norm_means, norm_stds = _fit_normalisation_stats(train_records, impute_fn, train_means)

        for split_name, records in [
            ("train", train_records), ("val", val_records), ("test", test_records)
        ]:
            out_dir = os.path.join(OUTPUT_ROOT, cond_name, split_name)
            _process_split(split_name, records, impute_fn, train_means,
                            norm_means, norm_stds, out_dir)

        # persist the fitted stats for later fine-tuning / inference-time use
        stats_path = os.path.join(OUTPUT_ROOT, cond_name, "norm_stats.npz")
        np.savez(stats_path,
                 train_means=train_means, norm_means=norm_means, norm_stds=norm_stds)
        print(f"Saved normalisation stats to {stats_path}")


if __name__ == "__main__":
    main()
