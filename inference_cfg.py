from pathlib import Path

GET_CSV = False

DATA_DIR = Path("/mnt/an225/Anna/longitudinal_nifti_BCH")
BASE_DIR_NO_OPS = DATA_DIR / "curated_no_ops"
BASE_DIR_MIXED = DATA_DIR / "curated_nifti_data"
BASE_DIR_DGM = DATA_DIR / "curated_dmg"
BASE_DIR_RADART = DATA_DIR / "curated_radart"
BASE_DIR_ADDITIONAL = DATA_DIR / "curated_no_ops_later_surgery"

BASE_OUTPUT = Path("/home/jc053/GIT/mri-sequence-classification/data_csv/long/")
OUTPUT_DIR_NO_OPS = BASE_OUTPUT / "curated_no_ops"
OUTPUT_DIR_MIXED = BASE_OUTPUT / "curated_nifti_data"
OUTPUT_DIR_DGM = BASE_OUTPUT / "curated_dgm"
OUTPUT_DIR_RADART = BASE_OUTPUT / "curated_radart"
OUTPUT_DIR_ADDITIONAL = BASE_OUTPUT / "curated_no_ops_later_surgery"


MODELS_DIR = Path("/home/jc053/GIT/mri-sequence-classification/models")
