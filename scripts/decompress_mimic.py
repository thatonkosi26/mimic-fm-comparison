"""
decompress_mimic.py

One-time utility: MIMIC-III is distributed as .csv.gz. The standard
mimic3-benchmarks extraction scripts (extract_subjects.py etc.) read
plain .csv files, so run this once before Step 1 in README.md.

Usage:
    python scripts/decompress_mimic.py
(reads config.RAW_MIMIC_GZ_DIR, writes to config.RAW_MIMIC_CSV_DIR)

This only needs to be run once. It skips files that are already
decompressed, so it's safe to re-run if it's interrupted partway
(the large tables -- CHARTEVENTS, LABEVENTS -- are multiple GB each
and will take a while).
"""

import gzip
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_MIMIC_GZ_DIR, RAW_MIMIC_CSV_DIR


def decompress_all():
    os.makedirs(RAW_MIMIC_CSV_DIR, exist_ok=True)
    gz_files = [f for f in os.listdir(RAW_MIMIC_GZ_DIR) if f.endswith(".csv.gz")]
    if not gz_files:
        print(f"No .csv.gz files found in {RAW_MIMIC_GZ_DIR} -- check the path.")
        return

    for fname in sorted(gz_files):
        out_name = fname[:-3]  # strip ".gz"
        src = os.path.join(RAW_MIMIC_GZ_DIR, fname)
        dst = os.path.join(RAW_MIMIC_CSV_DIR, out_name)
        if os.path.exists(dst):
            print(f"  skip (exists): {out_name}")
            continue
        print(f"  decompressing: {fname} -> {out_name}")
        with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"\nDone. Decompressed CSVs are in {RAW_MIMIC_CSV_DIR}")
    print("Point mimic3-benchmarks' extract_subjects.py at this folder.")


if __name__ == "__main__":
    decompress_all()
