"""
config.py
Shared constants for the RQ1-4 preprocessing pipeline described in
Chapter 3 of the proposal. Edit BENCHMARK_ROOT to point at the output
of the standard Harutyunyan et al. (2019) mimic3-benchmarks repo
(i.e. the folder containing train/ and test/ with *_timeseries.csv files
and listfile.csv).
"""

import os

# ---- project root (so preprocessing/, models/, etc. can all import this
# file regardless of which directory a script is launched from) ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- raw MIMIC-III source files ----
# This is where your downloaded .csv.gz files live. The mimic3-benchmarks
# extraction scripts (Step 1 in README.md) read PRESCRIPTIONS.csv,
# CHARTEVENTS.csv, etc. as PLAIN .csv, not .csv.gz, so these need to be
# decompressed once before running extract_subjects.py. See the note in
# README.md's "Windows / gzip" section.
RAW_MIMIC_GZ_DIR = r"C:\Users\thato\Downloads\mimic-iii-clinical-database-1.4\mimic-iii-clinical-database-1.4"
RAW_MIMIC_CSV_DIR = r"C:\Users\thato\mimic-iii-decompressed"  # output of the one-time gunzip step

# ---- mimic3-benchmarks output (Step 1) / this pipeline's input ----
BENCHMARK_ROOT = r"C:\Users\thato\Downloads\mimic3-benchmarks\data\in-hospital-mortality"
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")

TRAIN_DIR = os.path.join(BENCHMARK_ROOT, "train")
TEST_DIR = os.path.join(BENCHMARK_ROOT, "test")
TRAIN_LISTFILE = os.path.join(BENCHMARK_ROOT, "train_listfile.csv")
VAL_LISTFILE = os.path.join(BENCHMARK_ROOT, "val_listfile.csv")
TEST_LISTFILE = os.path.join(BENCHMARK_ROOT, "test_listfile.csv")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---- the 17 Harutyunyan variables, exactly as they appear as columns
# in the benchmark repo's timeseries CSVs ----
VARIABLES = [
    "Capillary refill rate",
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
]

# categorical GCS columns need label-encoding before imputation/normalisation
CATEGORICAL_VARS = [
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale verbal response",
]

WINDOW_HOURS = 48
IMPUTATION_CONDITIONS = ["forward_fill", "linear_interp"]  # Condition A, B

RANDOM_SEED = 42
