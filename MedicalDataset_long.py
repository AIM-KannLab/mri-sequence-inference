import os
import random
import time

import cv2
import nibabel as nib
import numpy
import pandas
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from SimpleITK import ImageSeriesReader
from torch.utils.data import Dataset

from time_util import time_format


class MedicalDataset(Dataset):
    def __init__(
        self,
        dataframe,
        min_slices=10,
    ):
        self.min_slices = min_slices
        self.images = dataframe.copy()
        self.original_indexes = (
            dataframe.index.to_list()
        )  # Keep track of original indexes
        self.loaded_data = self.load_data()

    def load_data(self):
        data = []
        start = time.time()
        counter = 0
        pixel_data = None

        for i, row in self.images.iterrows():
            counter += 1
            image_path = row["Path"]
            print("Loading", counter, "/", len(self.images), image_path)

            if (
                row["Path"].endswith((".bvec", ".bval"))
                and row["Prediction"] == "NO PREDICTION - METADATA"
            ):
                print("Already flagged row. Skipping", image_path)
                data.append(None)
                continue

            try:
                img = nib.load(image_path)
                spacing = img.header.get_zooms()
                orientation = nib.aff2axcodes(img.affine)
                shape = img.shape

                image = sitk.ReadImage(image_path)
                # spacing = str(image.GetSpacing())  # Store the ITK voxel spacing
                self.images.loc[
                    self.images["Path"] == image_path, "Image Spacing (x,y,z)"
                ] = str(spacing[:3])
                self.images.loc[
                    self.images["Path"] == image_path, "Image Orientation (Anatomical)"
                ] = str(orientation)
                self.images.loc[
                    self.images["Path"] == image_path, "Image Dimensions (x,y,z)"
                ] = str(shape)
            except RuntimeError:
                print("Skipping a non readbale image due to metadata", image_path)
                self.images.loc[
                    self.images["Path"] == image_path, "Prediction"
                ] = "NO PREDICTION - FILE ERROR "
                data.append(None)
                continue

            pixel_data, color_channels, direction = (
                self.normalize(sitk.GetArrayFromImage(image)),
                image.GetNumberOfComponentsPerPixel(),
                image.GetDirection(),
            )

            if color_channels > 1:
                pixel_data = numpy.array(
                    [cv2.cvtColor(slice, cv2.COLOR_RGB2GRAY) for slice in pixel_data]
                )

            # Ensure pixel_data has exactly 3 dimensions
            if pixel_data.ndim == 4:
                # If the image has a singleton dimension, squeeze it out
                if 1 in pixel_data.shape:
                    pixel_data = numpy.squeeze(pixel_data)
                else:
                    print("Skipping a non 3D convertable image", image_path)
                    self.images.loc[
                        self.images["Path"] == image_path, "Prediction"
                    ] = "NO PREDICTION - DIMS"
                    data.append(None)
                    continue
            elif pixel_data.ndim > 4:
                print("Skipping a 5D image", image_path)
                self.images.loc[
                    self.images["Path"] == image_path, "Prediction"
                ] = "NO PREDICTION - DIMS"
                data.append(None)
                continue

            n_slices = pixel_data.shape[0]
            min_slices = 16  # self.min_slices  # was harcore 16 before
            if n_slices < min_slices:
                extended = numpy.zeros(
                    (min_slices, pixel_data.shape[1], pixel_data.shape[2])
                ).astype(numpy.uint8)
                extended[
                    min_slices // 2
                    - n_slices // 2 : min_slices // 2
                    + n_slices // 2
                    + n_slices % 2,
                    :,
                    :,
                ] = pixel_data.copy()
                for j in range(min_slices // 2 - n_slices // 2):
                    extended[j] = pixel_data[0]
                for j in range(min_slices // 2 + n_slices // 2 + 1, min_slices):
                    extended[j] = pixel_data[-1]
                pixel_data = extended
            else:
                pixel_data = pixel_data[
                    n_slices // 2
                    - min_slices // 2 : n_slices // 2
                    + min_slices // 2
                    + min_slices % 2
                ]

            height, width = pixel_data.shape[1], pixel_data.shape[2]
            if height != width:
                min_dim = min(height, width)
                max_dim = max(height, width)
                zoom_out = numpy.zeros(pixel_data.shape).astype(numpy.uint8)
                if min_dim != 0:
                    ratio = min_dim / max_dim
                    for i, slice in enumerate(pixel_data):
                        if (
                            slice.shape[0] > 0 and slice.shape[1] > 0
                        ):  # check if the slice has a valid shape
                            if slice.shape[0] > slice.shape[1]:  # if height > width
                                new_dim = (min_dim, int(slice.shape[0] * ratio))
                            else:  # if width >= height
                                new_dim = (int(slice.shape[1] * ratio), min_dim)
                            slice = cv2.resize(slice, new_dim)
                            zoom_out[i, : slice.shape[0], : slice.shape[1]] = slice
                        else:
                            print(f"Skipping resize for slice {i} due to zero shape.")
                else:
                    print(
                        f"Skipping slice {image_path} resizing due to shape {slice.shape}"
                    )
                    self.images.loc[
                        self.images["Path"] == image_path, "Prediction"
                    ] = "NO PREDICTION - SLICE ERROR"
                    data.append(None)
                    continue

                pixel_data = zoom_out
                pixel_data = pixel_data[:, :min_dim, :min_dim]
                assert pixel_data.shape[1] == pixel_data.shape[2]

            pixel_data = numpy.array(
                [cv2.resize(slice, (200, 200)) for slice in pixel_data]
            )

            assert pixel_data.shape == (min_slices, 200, 200)

            data.append(pixel_data)

        print("\nLoading time:", time_format(time.time() - start))
        return data

    def get_dataframe(self):
        return self.images

    def rotate(self, image):
        def cos(vector1, vector2):
            for axis in range(0, 3):
                axes_relation = vector1[axis] * vector2[axis]
                if abs(axes_relation) == 1:
                    return axes_relation
            return 0

        direction = image.GetDirection()
        sagittal = (
            round(direction[0]),
            round(direction[1]),
            round(direction[2]),
        )  # Width, 2 (Lateral)
        coronal = (
            round(direction[3]),
            round(direction[4]),
            round(direction[5]),
        )  # Height, 1 (Front-Back)
        axial = (
            round(direction[6]),
            round(direction[7]),
            round(direction[8]),
        )  # Layers, 0   (Axial)

        coords_are_left_hand = (axial == numpy.cross(sagittal, coronal)).all()

        vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        random.shuffle(vectors)
        x, y, z = vectors[0], vectors[1], vectors[2]
        new_coords_are_left_hand = not coords_are_left_hand
        while new_coords_are_left_hand != coords_are_left_hand:
            x = tuple(random.choice([-1, 1]) * numpy.array(x))
            y = tuple(random.choice([-1, 1]) * numpy.array(y))
            z = tuple(random.choice([-1, 1]) * numpy.array(z))
            new_coords_are_left_hand = (z == numpy.cross(x, y)).all()

        rotation_matrix = numpy.array(
            [
                [cos(sagittal, x), cos(coronal, x), cos(axial, x)],
                [cos(sagittal, y), cos(coronal, y), cos(axial, y)],
                [cos(sagittal, z), cos(coronal, z), cos(axial, z)],
            ]
        ).astype(numpy.double)

        rotation_transform = sitk.AffineTransform(3)
        rotation_transform.SetMatrix(rotation_matrix.ravel())

        color_channels = image.GetNumberOfComponentsPerPixel()
        pixel_type = sitk.sitkFloat32 if color_channels == 1 else sitk.sitkVectorFloat32
        return sitk.Resample(image, rotation_transform, sitk.sitkLinear, 0, pixel_type)

    def normalize(self, pixel_data):
        pixel_data = pixel_data.astype(numpy.float32)
        min = numpy.min(pixel_data)
        max = numpy.max(pixel_data)

        pixel_data = numpy.uint8(255 * (pixel_data - min) / (max - min + 1e-10))

        return pixel_data

    def rotate_RAS(self, image):
        for axes in [(1, 0)]:  # [(0, 1), (0, 2), (1, 2)]:
            for n_rots in range(random.choice(range(4))):
                image = numpy.rot90(image, axes=axes)

        height, width = image.shape[0], image.shape[1]
        yroll, xroll = random.randint(-height // 10, height // 10), random.randint(
            -width // 10, width // 10
        )
        image = numpy.roll(image, yroll, axis=0)
        image = numpy.roll(image, xroll, axis=1)

        if yroll < 0:
            image[yroll:, :] = 0
        else:
            image[:yroll, :] = 0

        if xroll < 0:
            image[:, xroll:] = 0
        else:
            image[:, :xroll] = 0

        return image

    def transform(self, image):
        if self.min_slices > 1:
            image = numpy.transpose(image, (1, 2, 0))  # HWC
        else:
            image = image[0]

        if self.min_slices > 1:
            image = numpy.transpose(image, (2, 0, 1))
        else:
            image = numpy.array([image])

        aug_tensor = torch.tensor(image.astype(numpy.float32))
        aug_tensor = TF.normalize(
            aug_tensor, tuple(image.shape[0] * [0]), tuple(image.shape[0] * [1])
        )
        return aug_tensor

    def __getitem__(self, idx):
        image = self.loaded_data[idx]
        path, _, _, _, _ = self.images.iloc[idx]

        if image is None:
            # Return a special value that you can check for later
            return (None, path)

        first_slice_idx = numpy.random.randint(16 - self.min_slices + 1)
        last_slice_idx = first_slice_idx + self.min_slices
        image = image[first_slice_idx:last_slice_idx]

        return self.transform(image), path

    def __len__(self):
        return len(self.loaded_data)
