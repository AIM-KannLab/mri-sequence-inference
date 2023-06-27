import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import os
import nibabel as nib
import matplotlib.pyplot as plt


def nii_to_png(nii_path, output_path):
    # Load .nii.gz file
    nii_img = nib.load(nii_path)

    # Convert .nii.gz image data to numpy array
    data = nii_img.get_fdata()

    # Depending on the image you might want to access different index
    # Normally these images have 3 dimensions and to convert it to 2D you can take a slice of it
    slice_2d = data[:, :, data.shape[2] // 2]

    # Display the image slice
    plt.imshow(slice_2d, cmap="gray")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)


# Define csv_file
csv_file = "data_csv/long/curated_no_ops/OTHER/annotations.csv"
folder = "data_csv/long/curated_no_ops/OTHER/"

# Define column names
column_names = ["Image_Name", "Quality", "Comments"]

# Load the data
df = pd.read_csv(csv_file, names=column_names)

# Summary of the data
print(df.describe(include="all"))
# Convert Quality to integers
df["Quality"] = df["Quality"].astype(int)

# Compute some statistics
num_images = df["Image_Name"].nunique()
avg_quality = df["Quality"].mean()
quality_dist = df["Quality"].value_counts()

# Print the statistics
print(f"Number of unique images: {num_images}")
print(f"Average quality: {avg_quality:.2f}")
print("\nQuality distribution:")
print(quality_dist)

# Get all unique comments
all_comments = df["Comments"].unique()
print(f"All types of annotations: {all_comments}")

# Filter comments containing the word "flair" (case insensitive)
flair_comments = df[df["Comments"].str.contains("flair", case=False, na=False)]

# Count of comments containing the word "flair"
flair_count = len(flair_comments)
print(f"Number of comments containing 'flair': {flair_count}")

# Histogram of the quality ratings
plt.figure(figsize=(10, 6))
quality_plot = sns.countplot(data=df, x="Quality")
plt.title("Distribution of Quality Ratings")
# Add counts on the bars
for p in quality_plot.patches:
    quality_plot.annotate(
        format(p.get_height(), ".0f"),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )
plt.savefig("output_evaluation_t2.png")


# Generate a collage of image, quality and comment
# for index, row in df.iterrows():
#     try:
#         nii_path = os.path.join(folder, row["Image_Name"])
#         png_path = f"tmp_{index}.png"
#         nii_to_png(nii_path, png_path)

#         # Open the image file
#         with Image.open(png_path) as img:
#             width, height = img.size

#             # Create a new image with white background
#             background = Image.new("RGB", (width, height + 60), (255, 255, 255))
#             background.paste(img, (0, 60))

#             # Prepare to draw text
#             draw = ImageDraw.Draw(background)
#             # Load a font (you might need to change the font path depending on your system)
#             font = ImageFont.truetype(
#                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=15
#             )

#             # Draw text
#             draw.text((10, 10), f"Quality: {row['Quality']}", fill="black", font=font)
#             draw.text((10, 30), f"Comment: {row['Comments']}", fill="black", font=font)

#             # Save the image
#             background.save(f"output_{index}.png")

#     except Exception as e:
#         print(f"Unable to process image {row['Image_Name']}. Error: {e}")
