"""
verify_benchmark_output.py

Run this AFTER Step 1 (mimic3-benchmarks extraction) and BEFORE
build_dataset.py. It checks:
  1. The expected folders/listfiles exist at config.BENCHMARK_ROOT.
  2. The listfiles have the expected columns.
  3. A sample timeseries CSV's columns actually match config.VARIABLES --
     this is the most common silent-failure point, since a column-name
     mismatch doesn't raise an error, it just produces all-NaN features.

Usage:
    python scripts/verify_benchmark_output.py
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BENCHMARK_ROOT, TRAIN_DIR, TEST_DIR,
    TRAIN_LISTFILE, VAL_LISTFILE, TEST_LISTFILE, VARIABLES,
)


def check_paths_exist():
    print("=== Checking paths ===")
    ok = True
    for name, path in [
        ("BENCHMARK_ROOT", BENCHMARK_ROOT),
        ("TRAIN_DIR", TRAIN_DIR),
        ("TEST_DIR", TEST_DIR),
        ("TRAIN_LISTFILE", TRAIN_LISTFILE),
        ("VAL_LISTFILE", VAL_LISTFILE),
        ("TEST_LISTFILE", TEST_LISTFILE),
    ]:
        exists = os.path.exists(path)
        print(f"  [{'OK' if exists else 'MISSING'}] {name}: {path}")
        ok = ok and exists
    return ok


def check_listfile_columns():
    print("\n=== Checking listfile columns ===")
    ok = True
    for name, path in [("train", TRAIN_LISTFILE), ("val", VAL_LISTFILE), ("test", TEST_LISTFILE)]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        cols = list(df.columns)
        print(f"  {name}_listfile.csv columns: {cols}")
        if "stay" not in cols and cols[0] != "stay":
            print(f"    WARNING: expected a 'stay' column (or first column = filename), got {cols[0]}")
            ok = False
        if "y_true" not in cols and len(cols) < 2:
            print("    WARNING: expected a 'y_true' (or second) column with the mortality label")
            ok = False
        else:
            label_col = "y_true" if "y_true" in cols else cols[1]
            rate = df[label_col].mean()
            print(f"    mortality rate in {name}: {rate:.3f} "
                  f"({'OK, in expected 10-15% range-ish' if 0.05 < rate < 0.25 else 'CHECK THIS - looks off'})")
    return ok


def check_timeseries_columns():
    print("\n=== Checking a sample timeseries.csv against config.VARIABLES ===")
    if not os.path.exists(TRAIN_DIR):
        print("  TRAIN_DIR missing, skipping.")
        return False

    sample_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith("_timeseries.csv")]
    if not sample_files:
        print(f"  No *_timeseries.csv files found in {TRAIN_DIR}")
        return False

    sample_path = os.path.join(TRAIN_DIR, sample_files[0])
    df = pd.read_csv(sample_path)
    actual_cols = set(df.columns)
    expected_cols = set(VARIABLES)

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols - {"Hours"}

    print(f"  Sample file: {sample_files[0]}")
    print(f"  Columns present: {sorted(actual_cols)}")
    if missing:
        print(f"\n  *** MISMATCH *** These variables in config.VARIABLES were NOT found "
              f"in the actual CSV columns:\n      {sorted(missing)}")
        print("  Fix: update config.VARIABLES to match the actual column names exactly "
              "(check for whitespace/capitalization differences), or build_dataset.py "
              "will silently produce all-NaN data for these.")
    else:
        print("  All expected variables found. Good to proceed.")

    if extra:
        print(f"\n  Note: CSV has extra columns not in config.VARIABLES (likely fine, "
              f"e.g. row id columns): {sorted(extra)}")

    return len(missing) == 0


if __name__ == "__main__":
    paths_ok = check_paths_exist()
    if not paths_ok:
        print("\nFix the missing paths above before continuing.")
        sys.exit(1)
    check_listfile_columns()
    cols_ok = check_timeseries_columns()
    print("\n=== Summary ===")
    print("Safe to run build_dataset.py" if cols_ok else
          "DO NOT run build_dataset.py yet -- fix the column mismatch above first.")
