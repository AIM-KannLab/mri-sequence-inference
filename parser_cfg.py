from pathlib import Path

# BASE_DIR = directory where all of the csv. files of the dataset have been generated and inferred to
BASE_DIR = Path("/home/jc053/GIT/mri-sequence-classification/data_csv/long")
DIR_NO_OPS = BASE_DIR / "curated_no_ops"
DIR_MIXED = BASE_DIR / "curated_nifti_data"
DIR_RADART = BASE_DIR / "curated_radart"
DIR_DGM = BASE_DIR / "curated_dgm"
DIR_NO_OPS_ADD = BASE_DIR / "curated_no_ops_later_surgery"

OUTPUT_DIR = Path("/mnt/93E8-0534/JuanCarlos/mri-classification-sequences/")
OUT_DGM = OUTPUT_DIR / "curated_dgm_filtered"
OUT_NO_OPS = OUTPUT_DIR / "curated_no_ops"
OUT_NIFTI = OUTPUT_DIR / "curated_nifti_data"
OUT_RADART = OUTPUT_DIR / "curated_radart"
OUT_NO_OPS_ADD = OUTPUT_DIR / "curated_no_ops_later_surgery"
