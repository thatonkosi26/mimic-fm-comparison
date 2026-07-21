"""
imputation.py

Implements the two parallel preprocessing conditions from Section 3.3:

Condition A (Harutyunyan-style): resample to an hourly grid, forward-fill,
    then mean-impute any still-missing leading values using TRAINING SET
    statistics only.
Condition B (linear interpolation): resample to an hourly grid, linearly
    interpolate between observed values, then mean-impute any leading/
    trailing gaps using TRAINING SET statistics only.

Both functions operate on a single episode at a time and take the
training-set means as an argument, so the same code is reused for
train/val/test as long as the caller always passes in means computed
from the training split (see build_dataset.py).
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VARIABLES, CATEGORICAL_VARS, WINDOW_HOURS


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """GCS fields arrive as ordinal strings in the raw benchmark CSVs
    (e.g. '4 Spontaneously'). Map them to their leading integer code.
    Unparseable / missing entries become NaN so imputation handles them
    normally."""
    df = df.copy()
    for col in CATEGORICAL_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            )
    return df


def load_episode(path: str) -> pd.DataFrame:
    """Load one *_timeseries.csv from the benchmark repo output."""
    df = pd.read_csv(path)
    df = _encode_categoricals(df)
    # 'Hours' column gives time since ICU admission in fractional hours
    df["hour_bin"] = df["Hours"].apply(lambda h: int(np.floor(h)))
    df = df[(df["hour_bin"] >= 0) & (df["hour_bin"] < WINDOW_HOURS)]
    return df


def _to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse an episode's raw (possibly multiple-per-hour) observations
    onto a fixed 0..WINDOW_HOURS-1 hourly grid, taking the mean of any
    values that land in the same hour."""
    grid = pd.DataFrame(index=range(WINDOW_HOURS))
    for var in VARIABLES:
        if var in df.columns:
            hourly = df.groupby("hour_bin")[var].mean()
            grid[var] = hourly.reindex(range(WINDOW_HOURS))
        else:
            grid[var] = np.nan
    return grid


def condition_a_forward_fill(df: pd.DataFrame, train_means: dict) -> tuple:
    """Returns (values_df, mask_df) — mask_df marks which hourly cells were
    ORIGINALLY observed (1) vs imputed (0), needed for the missingness
    indicator features used by the traditional ML models (Sec 3.5.1)."""
    grid = _to_hourly_grid(df)
    mask = grid.notna().astype(int)
    grid = grid.ffill()
    for var in VARIABLES:
        grid[var] = grid[var].fillna(train_means[var])
    return grid, mask


def condition_b_linear_interp(df: pd.DataFrame, train_means: dict) -> tuple:
    grid = _to_hourly_grid(df)
    mask = grid.notna().astype(int)
    grid = grid.interpolate(method="linear", limit_direction="both", axis=0)
    for var in VARIABLES:
        grid[var] = grid[var].fillna(train_means[var])
    return grid, mask


def compute_training_means(episode_paths: list) -> dict:
    """Population means per variable, computed ONLY from training episodes,
    used as the fallback fill value in both conditions (Sec 3.3) and as
    the source for z-score normalisation (Sec 3.3, last paragraph)."""
    sums = {v: 0.0 for v in VARIABLES}
    counts = {v: 0 for v in VARIABLES}
    for p in episode_paths:
        df = load_episode(p)
        for v in VARIABLES:
            if v in df.columns:
                vals = df[v].dropna()
                sums[v] += vals.sum()
                counts[v] += len(vals)
    return {v: (sums[v] / counts[v] if counts[v] > 0 else 0.0) for v in VARIABLES}
