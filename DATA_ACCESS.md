# Data Access

This repository contains **code only**. No MIMIC-III data, derived
features, or model checkpoints trained on patient data are included or
should ever be committed here — see `.gitignore`.

## Getting access to MIMIC-III yourself

1. Create a [PhysioNet](https://physionet.org/) account.
2. Complete the CITI Program course *"Data or Specimens Only Research"*
   (or the equivalent human subjects research training PhysioNet
   currently requires).
3. Sign the PhysioNet Credentialed Health Data Use Agreement for
   MIMIC-III.
4. Download MIMIC-III v1.4 from
   [physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/).

## Reproducing this project's results

Once you have your own copy of MIMIC-III:

1. Follow `README.md` Step 0 to decompress the `.csv.gz` files.
2. Follow `README.md` Step 1 to run the standard
   [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)
   cohort extraction and get the Harutyunyan et al. (2019) train/val/test
   split.
3. Point `config.py`'s `BENCHMARK_ROOT` at that output.
4. Run `preprocessing/build_dataset.py`.

This mirrors the standard reproducibility convention in clinical ML: the
data use agreement is between you and PhysioNet, and the code here is
data-agnostic as long as it's pointed at a correctly-formatted MIMIC-III
extract.
