import argparse
import pandas as pd
import os
import shutil
from parser_long_cfg import (
    DIR_DGM,
    DIR_MIXED,
    DIR_NO_OPS,
    DIR_RADART,
    OUT_DGM,
    OUT_NIFTI,
    OUT_NO_OPS,
    OUT_RADART,
    DIR_NO_OPS_ADD,
    OUT_NO_OPS_ADD,
)
import gc
import sys

parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument(
    "--start-file",
    type=str,
    help="The file to start processing from",
    # default="2305625_file_paths.csv",
)

args = parser.parse_args()

start_file = args.start_file


def extract_ids_from_path(path):
    ids = path.split("/")[-3:-1]
    return ids


# EDIT THIS TO PROCESS ALL OR SINGLE FOLDERS
main_directories = [
    DIR_NO_OPS_ADD,
]
output_directory_mapping = {
    DIR_DGM: OUT_DGM,
    DIR_RADART: OUT_RADART,
    DIR_MIXED: OUT_NIFTI,
    DIR_NO_OPS: OUT_NO_OPS,
    DIR_NO_OPS_ADD: OUT_NO_OPS_ADD,
}

sub_directories = ["FLAIR", "T1", "T1c", "T2", "OTHER", "NO PREDICTION"]
ignore_strings = [
    "spine",
    "ctl",
    "lumbar",
    "tlsp",
    "tsp",
    "lsp",
    "csp",
    "t1_flair",
    "stir",
    "left",
    "right",
    "dti",
    "dwi",
]

for main_dir in main_directories:
    for sub_dir in sub_directories:
        output_dir = output_directory_mapping[main_dir]
        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)

    csv_files = [f for f in os.listdir(main_dir) if f.endswith(".csv")]

    file_started = False if start_file else True
    for file_name in csv_files:
        if not file_started:
            if file_name == start_file:
                file_started = True
            else:
                continue
        print(f"\tProcessing file: {file_name}")
        for chunk in pd.read_csv(
            os.path.join(main_dir, file_name), chunksize=10000
        ):  # adjust chunksize based on your system's memory
            chunk["Image Spacing (x,y,z)"] = chunk["Image Spacing (x,y,z)"].apply(
                lambda x: eval(x) if pd.notnull(x) else (0, 0, 0)
            )
            chunk["Image Dimensions (x,y,z)"] = chunk["Image Dimensions (x,y,z)"].apply(
                lambda x: eval(x) if pd.notnull(x) else (0, 0, 0)
            )

            chunk["Spacing_X"], chunk["Spacing_Y"], _ = zip(
                *chunk["Image Spacing (x,y,z)"]
            )
            chunk["Dimension_X"], chunk["Dimension_Y"], _ = zip(
                *chunk["Image Dimensions (x,y,z)"]
            )

            mask = (
                (chunk["Spacing_X"] < 2.0)
                & (chunk["Spacing_Y"] < 2.0)
                & (chunk["Dimension_X"] >= 256)
                & (chunk["Dimension_Y"] >= 256)
            )

            for prediction in sub_directories:
                if prediction != "NO PREDICTION":
                    df_filtered = chunk[mask & (chunk["Prediction"] == prediction)]
                    if not df_filtered.empty:
                        df_filtered.to_csv(
                            os.path.join(output_dir, prediction, file_name), index=False
                        )
                        for _, row in df_filtered.iterrows():
                            assert os.path.exists(row["Path"])
                            if not any(
                                ignore_string in row["Path"]
                                for ignore_string in ignore_strings
                            ):
                                patient_id, scan_id = extract_ids_from_path(row["Path"])
                                dest_path = os.path.join(
                                    output_dir,
                                    prediction,
                                    f'{patient_id}_{scan_id}_{os.path.basename(row["Path"])}',
                                )
                                if not os.path.exists(dest_path):
                                    try:
                                        shutil.copyfile(row["Path"], dest_path)
                                    except IOError as e:
                                        print(f"Unable to copy file. {e}")
                                    except:
                                        print("Unexpected error:", sys.exc_info())

            no_prediction_df = chunk[
                ~mask | chunk["Prediction"].str.startswith("NO PREDICTION")
            ]
            if not no_prediction_df.empty:
                no_prediction_df.to_csv(
                    os.path.join(output_dir, "NO PREDICTION", file_name), index=False
                )
                for _, row in no_prediction_df.iterrows():
                    assert os.path.exists(row["Path"])
                    if not any(
                        ignore_string in row["Path"] for ignore_string in ignore_strings
                    ):
                        patient_id, scan_id = extract_ids_from_path(row["Path"])
                        dest_path = os.path.join(
                            output_dir,
                            "NO PREDICTION",
                            f'{patient_id}_{scan_id}_{os.path.basename(row["Path"])}',
                        )
                        if not os.path.exists(dest_path):
                            try:
                                shutil.copyfile(row["Path"], dest_path)
                            except IOError as e:
                                print(f"Unable to copy file. {e}")
                            except:
                                print("Unexpected error:", sys.exc_info())

            del chunk
            gc.collect()
