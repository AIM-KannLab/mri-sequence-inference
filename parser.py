"""
Parser for the longitudinal dataset. Moves files to the classified folders.
"""
import argparse
import gc
import os
import shutil

import pandas as pd

from parser_long_cfg import IN_DIR, OUT_DIR
import ast

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
    """
    Extracts the patient and scan ids from the path.
    """
    ids = path.split("/")[-3:-1]
    return ids


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

for sub_dir in sub_directories:
    output_dir = OUT_DIR
    os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)

csv_files = [f for f in os.listdir(IN_DIR) if f.endswith(".csv")]

file_started = False if start_file else True
for file_name in csv_files:
    if not file_started:
        if file_name == start_file:
            file_started = True
        else:
            continue
    print(f"\tProcessing file: {file_name}")
    for chunk in pd.read_csv(
        os.path.join(IN_DIR, file_name), chunksize=10000
    ):  # adjust chunksize based on your system's memory
        chunk["Image Spacing (x,y,z)"] = chunk["Image Spacing (x,y,z)"].apply(
            lambda x: ast.literal_eval(x) if pd.notnull(x) else (0, 0, 0)
        )
        chunk["Image Dimensions (x,y,z)"] = chunk["Image Dimensions (x,y,z)"].apply(
            lambda x: ast.literal_eval(x) if pd.notnull(x) else (0, 0, 0)
        )

        chunk["Spacing_X"], chunk["Spacing_Y"], _ = zip(*chunk["Image Spacing (x,y,z)"])
        chunk["Dimension_X"], chunk["Dimension_Y"], _ = zip(*chunk["Image Dimensions (x,y,z)"])

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
                    df_filtered.to_csv(os.path.join(output_dir, prediction, file_name), index=False)
                    for _, row in df_filtered.iterrows():
                        assert os.path.exists(row["Path"])
                        if not any(
                            ignore_string in row["Path"] for ignore_string in ignore_strings
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
                                except SystemError as error:
                                    print("Unexpected error:", error)

        no_prediction_df = chunk[~mask | chunk["Prediction"].str.startswith("NO PREDICTION")]
        if not no_prediction_df.empty:
            no_prediction_df.to_csv(
                os.path.join(output_dir, "NO PREDICTION", file_name), index=False
            )
            for _, row in no_prediction_df.iterrows():
                assert os.path.exists(row["Path"])
                if not any(ignore_string in row["Path"] for ignore_string in ignore_strings):
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
                        except SystemError as error:
                            print("Unexpected error:", error)

        del chunk
        gc.collect()
