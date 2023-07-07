import argparse
import os
import random
import time
import pandas as pd
import numpy
import torch
import torch.utils.data as data
from torch.cuda.amp import autocast
import gc

from MedicalDataset_long import MedicalDataset
from models import select_net
from time_util import time_format
import os
import csv
from tqdm import tqdm
from config import (
    BASE_DIR_NO_OPS,
    OUTPUT_DIR_NO_OPS,
    GET_CSV,
    MODELS_DIR,
    BASE_DIR_MIXED,
    OUTPUT_DIR_MIXED,
    BASE_DIR_DGM,
    BASE_DIR_RADART,
    OUTPUT_DIR_DGM,
    OUTPUT_DIR_RADART,
    BASE_DIR_ADDITIONAL,
    OUTPUT_DIR_ADDITIONAL,
)
from statistics import mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        dest="test_data_path",
        type=str,
        help="Path containing data to be tested.",
        default=OUTPUT_DIR_ADDITIONAL,
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
    torch.backends.cudnn.deterministic = True
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    numpy.random.seed(1)


def get_files_and_write_to_csv(base_dir, output_dir):
    # check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop through patient_id directories
    for patient in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient)

        # check if it is a directory
        if os.path.isdir(patient_path):
            file_paths = []

            # loop through scan_id directories
            for scan in os.listdir(patient_path):
                scan_path = os.path.join(patient_path, scan)

                # check if it is a directory
                if os.path.isdir(scan_path):
                    # loop through files in files directory
                    for file in os.listdir(scan_path):
                        file_path = os.path.join(scan_path, file)

                        # check if it is a file, not a directory
                        if os.path.isfile(file_path):
                            file_paths.append(file_path)

            # if there are any files, write them to a CSV
            if file_paths:
                with open(
                    os.path.join(output_dir, f"{patient}_file_paths.csv"),
                    "w",
                    newline="",
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
                        writer.writerow({"Path": path, "Prediction": ""})


def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:  # All items were None
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def process_batch(i, batch, net, tridim, classes, updated_df):
    if batch is None:  # Skip if data is None
        return updated_df

    pixel_data, path = batch
    current_file_path = path[0]
    if tridim:
        pixel_data = pixel_data.view(-1, 1, 10, 200, 200)

    try:
        outputs = net(pixel_data.cuda())
        _, predicted = torch.max(outputs.data, 1)

        predicted_class = classes[predicted.cpu().numpy()[0]]
        print(f"For file {i} at {path[0]}, model predicts: {predicted_class}")

        updated_df.loc[
            updated_df["Path"] == current_file_path, "Prediction"
        ] = predicted_class

    except Exception as e:
        print(f"Error processing file {i} at {path[0]}: {str(e)}")
        updated_df.loc[
            updated_df["Path"] == current_file_path, "Prediction"
        ] = "PREDICTION ERROR"

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
                test_set, num_workers=1, pin_memory=True, collate_fn=custom_collate
            )

            classes = ["FLAIR", "T1", "T1c", "T2", "OTHER"]
            start_time = time.time()
            n_test_files = test_set.__len__()

            print("Number slices: ", n_slices)
            print("Number of volumes for inference", n_test_files)
            print("Length of dataframe", len(updated_df))

            with torch.no_grad():
                for i, batch in tqdm(
                    enumerate(test_loader),
                    total=len(test_loader),
                    desc=f"Predicting from {csv_file}",
                ):
                    updated_df = process_batch(
                        i, batch, net, tridim, classes, updated_df
                    )

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
            print(
                "Testing elapsed time for file {0}: {1}".format(csv_file, elapsed_time)
            )


if __name__ == "__main__":
    if GET_CSV:
        get_files_and_write_to_csv(BASE_DIR_ADDITIONAL, OUTPUT_DIR_ADDITIONAL)
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
