"""
Parser for the longitudinal dataset.
"""
from pathlib import Path

# BASE_DIR = directory where all of the csv. files of
# the dataset have been generated and inferred to
BASE_DIR = Path("/home/jc053/GIT/mri-sequence-classification/data_csv/long")
IN_DIR = BASE_DIR / "bch_longitudinal_dataset_filtered"

OUTPUT_DIR = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/")
OUT_DIR = OUTPUT_DIR / "bch_longitudinal_dataset_filtered"
