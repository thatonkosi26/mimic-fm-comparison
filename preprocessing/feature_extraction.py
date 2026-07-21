"""
feature_extraction.py

Builds the 102-dimensional static feature vector for logistic regression /
Random Forest / XGBoost, exactly as specified in Section 3.5.1:
  - 5 summary stats (mean, min, max, std, observation count) per variable
    -> 17 * 5 = 85 features
  - 1 missingness fraction per variable (fraction of the 48 hourly slots
    that were originally observed, from the mask returned by imputation.py)
    -> 17 features
  Total = 102
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VARIABLES, WINDOW_HOURS


def extract_static_features(values_df: pd.DataFrame, mask_df: pd.DataFrame) -> np.ndarray:
    """
    values_df, mask_df: the (48, 17) imputed grid and observation mask
    returned by condition_a_forward_fill / condition_b_linear_interp.
    Note: mean/min/max/std are computed here on the IMPUTED grid, matching
    what the deep learning models see, so all model families are trained on
    a consistent underlying representation of each patient's course.
    """
    feats = []
    for var in VARIABLES:
        col = values_df[var].to_numpy(dtype=float)
        feats.extend([
            np.mean(col),
            np.min(col),
            np.max(col),
            np.std(col),
            mask_df[var].sum(),          # raw observation count
        ])
    for var in VARIABLES:
        feats.append(mask_df[var].sum() / WINDOW_HOURS)   # missingness fraction
    return np.array(feats, dtype=float)


FEATURE_NAMES = (
    [f"{v}_{stat}" for v in VARIABLES for stat in ("mean", "min", "max", "std", "count")]
    + [f"{v}_obs_frac" for v in VARIABLES]
)
assert len(FEATURE_NAMES) == 102
