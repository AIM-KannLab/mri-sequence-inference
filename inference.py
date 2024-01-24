"""
Sequence inference script for the longitudinal images.
"""
import argparse
import csv
import gc
import os
import random
import time

import numpy
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm

from config import BASE_DIR, GET_CSV, ID_COLUMN, IDS_FILE, MODELS_DIR, OUTPUT_DIR, READ_IDS
from MedicalDataset_long import MedicalDataset
from models import select_net
from time_util import time_format


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        dest="test_data_path",
        type=str,
        help="Path containing data to be tested.",
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "-m",
        dest="model_file",
        type=str,
        help="Name of the trained model file.",
        default="main_sl10.pth",
    )
    parser.add_argument(
        "-sl",
        dest="slices",
        type=int,
        default=10,
        help="Number of central slices considered by the trained model.",
    )
    parser.add_argument(
        "-3d",
        dest="tridim",
        action="store_true",
        help="Use if the trained model used tridimensional convolution.",
    )
    parser.add_argument(
        "--net",
        dest="net",
        type=str,
        default="resnet18",
        help="Network architecture to be used.",
    )
    return parser.parse_args()


def fix_random_seeds():
    """
    Fix random seeds for reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    numpy.random.seed(1)


def get_files_and_write_to_csv(base_dir, output_dir, ids_2_check=None):
    """
    Given a base directory, loop through all patient_id directories
    and write the paths of all files to a CSV.
    """
    # check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # loop through patient_id directories
    for patient in os.listdir(base_dir):
        if ids_2_check is not None and patient not in ids_2_check:
            continue

        patient_path = os.path.join(base_dir, patient)
        if os.path.isdir(patient_path) and patient[0].isdigit():
            file_paths = []

            for scan in os.listdir(patient_path):
                scan_path = os.path.join(patient_path, scan)
                if os.path.isdir(scan_path):
                    for file in os.listdir(scan_path):
                        file_path = os.path.join(scan_path, file)
                        if os.path.isfile(file_path):
                            file_paths.append(file_path)

            # if there are any files, write them to a CSV
            if file_paths:
                with open(
                    os.path.join(output_dir, f"{patient}_file_paths.csv"),
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    fieldnames = [
                        "Path",
                        "Prediction",
                        "Image Spacing (x,y,z)",
                        "Image Orientation (Anatomical)",
                        "Image Dimensions (x,y,z)",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for path in file_paths:
                        writer.writerow({"Path": path})


def custom_collate(batch):
    """
    Custom collate function to filter out None values.
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:  # All items were None
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def process_batch(i, batch, net, tridim, classes, updated_df):
    """
    Batch processing function.
    """
    if batch is None:  # Skip if data is None
        return updated_df

    pixel_data, path = batch
    current_file_path = path[0]
    if tridim:
        pixel_data = pixel_data.view(-1, 1, 10, 200, 200)

    try:
        outputs = net(pixel_data.cuda())
        # pylint: disable=E1101
        _, predicted = torch.max(outputs.data, 1)

        predicted_class = classes[predicted.cpu().numpy()[0]]
        print(f"For file {i} at {path[0]}, model predicts: {predicted_class}")

        updated_df.loc[updated_df["Path"] == current_file_path, "Prediction"] = predicted_class

    except (IOError, FileNotFoundError) as error:
        print(f"Error processing file {i} at {path[0]}: {str(error)}")
        updated_df.loc[updated_df["Path"] == current_file_path, "Prediction"] = "PREDICTION ERROR"

    del pixel_data
    return updated_df


def perform_prognosis_on_csvs(
    test_data_path,
    model_file,
    n_slices,
    tridim,
    architecture,
    models_dir,
):
    """
    Main function to perform prognosis on a set of CSV files containing
    the paths to the images to be tested.
    """
    # Make sure the models_dir is a directory
    assert os.path.isdir(models_dir), f"{models_dir} is not a directory."
    assert os.path.isdir(test_data_path), f"{test_data_path} is not a directory."

    # Model
    net = select_net(architecture, n_slices, tridim, consider_other_class=True)
    net.load_state_dict(torch.load(os.path.join(models_dir, model_file)))
    net.eval()

    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)

    for csv_file in tqdm(os.listdir(test_data_path), desc="Processing CSVs"):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(test_data_path, csv_file)
            df = pd.read_csv(csv_path)

            # Check if 'Prediction' column is already completely filled
            if not df["Prediction"].isnull().any():
                print(f"Skipping {csv_file} as it has already been processed.")
                continue

            # Assign 'OTHER' to the 'Prediction' column where the 'Path' ends with .bvec or .bval
            df.loc[
                df["Path"].str.endswith((".bvec", ".bval")), "Prediction"
            ] = "NO PREDICTION - METADATA"

            fix_random_seeds()

            test_set = MedicalDataset(
                dataframe=df,
                min_slices=n_slices,
            )

            updated_df = test_set.get_dataframe()

            test_loader = data.DataLoader(
                test_set, batch_size=1, num_workers=1, pin_memory=True, collate_fn=custom_collate
            )

            classes = ["FLAIR", "T1", "T1c", "T2", "OTHER"]
            start_time = time.time()
            n_test_files = len(test_set)

            print("Number slices: ", n_slices)
            print("Number of volumes for inference", n_test_files)
            print("Length of dataframe", len(updated_df))

            with torch.no_grad():
                for i, batch in tqdm(
                    enumerate(test_loader),
                    total=len(test_loader),
                    desc=f"Predicting from {csv_file}",
                ):
                    updated_df = process_batch(i, batch, net, tridim, classes, updated_df)

            # Invoke garbage collection and CUDA memory clearing right after inner loop
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # save the data_frame with the new 'Prediction' values to the csv file
            updated_df.to_csv(csv_path, index=False)

            # time
            print()
            end_time = time.time()
            elapsed_time = time_format(end_time - start_time)
            print(f"Testing elapsed time for file {csv_file}: {elapsed_time}")


def check_processed_ids(output_dir, ids_2_check):
    """Comparing the processed ids to the ids_2_check"""
    created_csv_ids = set()
    for filename in os.listdir(output_dir):
        if filename.endswith(".csv"):
            patient_id = filename.split("_")[0]
            created_csv_ids.add(patient_id)

    for id_ in created_csv_ids:
        id_ = str(id_)
    # Converting ids_2_check to a set for comparison
    ids_2_check_set = set(ids_2_check)
    # Finding the difference
    unprocessed_ids = ids_2_check_set - created_csv_ids

    print(f"IDs from ids_2_check not processed: {unprocessed_ids}")


def prefix_zeros_to_six_digit_ids(patient_id):
    """
    Adds 0 to the beginning of 6-digit patient IDs.
    """
    str_id = str(patient_id)
    if len(str_id) == 6:
        # print(f"Found a 6-digit ID: {str_id}. Prefixing a '0'.")
        patient_id = "0" + str_id

    else:
        patient_id = str_id
    return patient_id


if __name__ == "__main__":
    if GET_CSV:
        if READ_IDS:
            df_ids = pd.read_csv(IDS_FILE)
            ids_to_check = df_ids[ID_COLUMN].tolist()
            ids_to_check = [prefix_zeros_to_six_digit_ids(id) for id in ids_to_check]
            print(f"IDs to check: {len(ids_to_check)}")
            check_processed_ids(OUTPUT_DIR, ids_to_check)

        get_files_and_write_to_csv(BASE_DIR, OUTPUT_DIR, ids_to_check)
    else:
        args = parse_args()
        perform_prognosis_on_csvs(
            args.test_data_path,
            args.model_file,
            args.slices,
            args.tridim,
            args.net,
            MODELS_DIR,
        )
