"""
test_pipeline_output.py

Run with: pytest tests/test_pipeline_output.py -v
(or just `python tests/test_pipeline_output.py` for a plain script run)

These check the OUTPUT of build_dataset.py, not the code itself --
i.e. run this after you've actually processed real MIMIC-III data.
They exist to catch the kind of bug that doesn't crash anything but
quietly invalidates results: wrong shapes, leaked test statistics,
all-NaN columns from a silent column-name mismatch, etc.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_ROOT, IMPUTATION_CONDITIONS, VARIABLES

SPLITS = ["train", "val", "test"]


def _load(cond, split, name):
    path = os.path.join(OUTPUT_ROOT, cond, split, f"{name}.npy")
    if not os.path.exists(path):
        pytest.skip(f"{path} not found -- run build_dataset.py first")
    return np.load(path, allow_pickle=True)


@pytest.mark.parametrize("cond", IMPUTATION_CONDITIONS)
@pytest.mark.parametrize("split", SPLITS)
def test_shapes_consistent(cond, split):
    seqs = _load(cond, split, "sequences")
    feats = _load(cond, split, "static_feats")
    labels = _load(cond, split, "labels")
    ids = _load(cond, split, "subject_ids")

    n = labels.shape[0]
    assert seqs.shape == (n, 48, len(VARIABLES)), f"sequences shape wrong: {seqs.shape}"
    assert feats.shape == (n, 102), f"static_feats shape wrong: {feats.shape}"
    assert ids.shape[0] == n


@pytest.mark.parametrize("cond", IMPUTATION_CONDITIONS)
@pytest.mark.parametrize("split", SPLITS)
def test_no_nans_remain(cond, split):
    seqs = _load(cond, split, "sequences")
    feats = _load(cond, split, "static_feats")
    assert not np.isnan(seqs).any(), (
        "NaNs found in sequences -- likely a column-name mismatch between "
        "config.VARIABLES and the actual benchmark CSVs. Run "
        "scripts/verify_benchmark_output.py to check."
    )
    assert not np.isnan(feats).any(), "NaNs found in static_feats"


@pytest.mark.parametrize("cond", IMPUTATION_CONDITIONS)
@pytest.mark.parametrize("split", SPLITS)
def test_mortality_rate_plausible(cond, split):
    labels = _load(cond, split, "labels")
    rate = labels.mean()
    assert 0.05 < rate < 0.25, (
        f"Mortality rate {rate:.3f} for {cond}/{split} is outside the "
        f"10-15%-ish range your proposal expects -- check cohort extraction."
    )


@pytest.mark.parametrize("cond", IMPUTATION_CONDITIONS)
def test_no_subject_overlap_across_splits(cond):
    """The single most important check: no patient should appear in more
    than one of train/val/test, or your test-set metrics are invalid."""
    train_ids = set(_load(cond, "train", "subject_ids").tolist())
    val_ids = set(_load(cond, "val", "subject_ids").tolist())
    test_ids = set(_load(cond, "test", "subject_ids").tolist())

    assert not (train_ids & val_ids), "Overlap between train and val subject IDs!"
    assert not (train_ids & test_ids), "Overlap between train and test subject IDs!"
    assert not (val_ids & test_ids), "Overlap between val and test subject IDs!"


@pytest.mark.parametrize("cond", IMPUTATION_CONDITIONS)
def test_normalisation_stats_saved(cond):
    """Confirms norm_stats.npz exists and was computed only from train
    data (Section 3.3/3.7's leakage requirement) -- this just checks the
    file is present and has the right keys; the actual train-only
    computation is enforced in build_dataset.py itself."""
    stats_path = os.path.join(OUTPUT_ROOT, cond, "norm_stats.npz")
    if not os.path.exists(stats_path):
        pytest.skip(f"{stats_path} not found -- run build_dataset.py first")
    stats = np.load(stats_path, allow_pickle=True)
    for key in ("train_means", "norm_means", "norm_stds"):
        assert key in stats, f"missing key {key} in norm_stats.npz"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
