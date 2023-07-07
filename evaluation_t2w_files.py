import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(sys.path.append(str(Path(__file__).resolve().parent.parent)))
from tqdm import tqdm

from cfg.evaluation_cfg import (
    CSV_FILE,
    DATA_FOLDER,
    DIR1_NO_COMMENTS,
    DIR1_WITH_COMMENTS,
    DIR5,
    MOVING,
    OUT_FILE,
)


# Define function to move files based on condition once review
def move_files(df, condition, source_dir, destination_folder):
    filenames = df[condition]["Image_Name"]
    for filename in tqdm(filenames, desc="Moving files"):
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_folder, filename)
        if os.path.isfile(source_path):
            shutil.move(source_path, destination_path)


# START SCRIPT: Define csv_file
csv_file = CSV_FILE

# Define column names and load
column_names = ["Image_Name", "Quality", "Comments"]
df = pd.read_csv(csv_file, names=column_names)

# Compute some statistics
df["Quality"] = df["Quality"].astype(int)
num_images = df["Image_Name"].nunique()
avg_quality = df["Quality"].mean()
quality_dist = df["Quality"].value_counts()

print(f"Number of unique images: {num_images}")
print(f"Average quality: {avg_quality:.2f}")
print("\nQuality distribution:", quality_dist)

# Get all unique comments
all_comments = df["Comments"].unique()
print(f"All types of annotations: {all_comments}")


# Filter comments for quality 5
quality5_comments = df[df["Quality"] == 5]["Comments"]
quality1_comments = df[df["Quality"] == 1]["Comments"]
print("\nComments for quality rating of 5:", quality5_comments.value_counts())
print("\nComments for quality rating of 1:", quality1_comments.value_counts())

# Number of images with quality 1
quality1_no_comments = df[(df["Quality"] == 1) & (df["Comments"].isna())]
quality1_with_comments = df[(df["Quality"] == 1) & (df["Comments"].notna())]
print(
    f"\nNumber of images with quality rating of 1 and no comments: {len(quality1_no_comments)}"
)
print(
    f"Number of images with quality rating of 1 and with comments: {len(quality1_with_comments)}"
)

# Number of images with quality 5
quality5_no_comments = df[(df["Quality"] == 5) & (df["Comments"].isna())]
quality5_with_comments = df[(df["Quality"] == 5) & (df["Comments"].notna())]
print(
    f"\nNumber of images with quality rating of 5 and no comments: {len(quality5_no_comments)}"
)
assert len(quality5_no_comments) == 0
print(
    f"Number of images with quality rating of 5 and with comments: {len(quality5_with_comments)}"
)

# Histogram of the quality
plt.figure(figsize=(10, 6))
quality_plot = sns.countplot(data=df, x="Quality")
plt.title("Distribution of Quality Ratings")

# Compute some statistics
df["Quality"] = df["Quality"].astype(int)
df["Comments"] = df["Comments"].fillna("None")

# Define comment categories
comment_categories = [
    "FLAIR",
    "T1",
    "T1c",
    "OTHER",
    "None",
    "artifact",
    "quality",
    "view",
    "cropped",
]
quality_list = [int(label.get_text()) for label in quality_plot.get_xticklabels()]
legend_patches = []

for i, p in enumerate(quality_plot.patches):
    # Get corresponding quality rating
    quality = quality_list[i]

    # Annotate total count
    quality_plot.annotate(
        format(p.get_height(), ".0f"),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )

    # Only calculate comment categories for quality ratings 1 and 5
    if quality in [1, 5]:
        # Get comments for current quality rating
        comments = df[df["Quality"] == quality]["Comments"]

        # Count comment categories
        comment_counts = Counter(comments)

        # Add to the legend patches list
        for category in comment_categories:
            count = comment_counts.get(category, 0)
            legend_patches.append(
                mpatches.Patch(
                    color="none", label=f"Quality {quality} - {category}: {count}"
                )
            )

# Create legend
plt.legend(handles=legend_patches, bbox_to_anchor=(1, 1), loc="upper left")

plt.tight_layout()
plt.savefig(OUT_FILE)


# Moving Files if needed
moving = MOVING
if moving:
    # Folder variables
    source_dir = DATA_FOLDER
    dir1_no_comments = DIR1_NO_COMMENTS
    dir1_with_comments = DIR1_WITH_COMMENTS
    dir5 = DIR5

    # Make dirs
    os.makedirs(dir1_no_comments, exist_ok=True)
    os.makedirs(dir1_with_comments, exist_ok=True)
    os.makedirs(dir5, exist_ok=True)

    # Move the files
    move_files(
        df,
        (df["Quality"] == 1) & (df["Comments"] == "None"),
        source_dir,
        dir1_no_comments,
    )
    move_files(
        df,
        (df["Quality"] == 1) & (df["Comments"] != "None"),
        source_dir,
        dir1_with_comments,
    )
    move_files(df, df["Quality"] == 5, source_dir, dir5)
