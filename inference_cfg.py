"""
Basic config file containining paths to data and output directories.
"""
from pathlib import Path

GET_CSV = False
READ_IDS = True
ID_COLUMN = "BCH MRN"


DATA_DIR = Path("/mnt/an225/Anna/longitudinal_nifti_BCH")
BASE_DIR = DATA_DIR / "curated_BCH"

BASE_OUTPUT = Path("/home/jc053/GIT/mri-sequence-classification/data_csv/long")
OUTPUT_DIR = BASE_OUTPUT / "bch_longitudinal_dataset"
IDS_FILE = OUTPUT_DIR / "ids.csv"

MODELS_DIR = Path("/home/jc053/GIT/mri-sequence-classification/models")
