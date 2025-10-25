import os
import numpy

from Preprocessing.generate_training_data_csv import generate_csv_file
from plantcv import plantcv as pcv

from Utilities.util import (
    create_directory_if_needed,
    save_image
)

from project_configuration import (
    All_Preprocessed_Training_Data_Directories
)

from Preprocessing.compute_all_training_data import (
    plant_village_tomato_leaf_image_directories,
    compute_all_training_data_info
)

def generate_all_training_data():
    # Generate CSV training file
    generate_csv_file()

    # Create all training data directories
    create_all_training_image_data_directories()

    # Preprocess training data and save to disk
    preprocess_training_data()

def create_all_training_image_data_directories():
    tomato_leaf_image_categories = plant_village_tomato_leaf_image_directories()

    for directory in All_Preprocessed_Training_Data_Directories:

        create_directory_if_needed(directory)

        for category in tomato_leaf_image_categories:
            tomato_leaf_image_category_directory = os.path.join(
                directory,
                category
            )

            create_directory_if_needed(tomato_leaf_image_category_directory)

def preprocess_training_data():
    training_data_info = compute_all_training_data_info()

    for datum_info in training_data_info:
        # Read original rgb image
        rgb_image, _, _ = pcv.readimage(datum_info.original_rgb_image_path, mode='rgb')

        # Grayscale conversion
        grayscale_image = transform_rgb_image_to_grayscale(rgb_image)
        save_image(destination=datum_info.grayscale_image_path, img=grayscale_image)

        # Equalize grayscale
        equalized_grayscale_image = transform_grayscale_image_to_histogram_equalized_image(
            grayscale_image=grayscale_image
        )
        save_image(destination=datum_info.equalized_grayscale_image_path, img=equalized_grayscale_image)

        # Median Filter
        median_filtered_image = transform_histogram_equalized_grayscale_image_to_median_filtered_image(
            equalized_histogram_grayscale_image=equalized_grayscale_image
        )
        save_image(destination=datum_info.median_filtered_equalized_grayscale_image_path, img=median_filtered_image)

        # Otsu segmentation binary mask
        otsu_binary_mask = transform_median_filtered_equalized_histogram_grayscale_image_to_otsu_segmented_image(
            median_filtered_equalized_histogram_grayscale_image=median_filtered_image
        )
        save_image(destination=datum_info.otsu_binary_mask_image_path, img=otsu_binary_mask)

        # Background substraction Removal (segmentation)
        segmented_image = segment_grayscale_image_with_otsu_binary_mask(
            otsu_binary_mask=otsu_binary_mask,
            grayscale_image=grayscale_image
        )
        save_image(destination=datum_info.otsu_segmented_image_path, img=segmented_image)

        # Calculate feature vector

def transform_rgb_image_to_grayscale(rgb_image: numpy.ndarray) -> numpy.ndarray:
    """
    Transforms RGB image to a grayscale image
    """
    grayscale_image = pcv.rgb2gray(rgb_img=rgb_image)
    return grayscale_image

def transform_grayscale_image_to_histogram_equalized_image(grayscale_image: numpy.ndarray) -> numpy.ndarray:
    """
    Transforms grayscale image at path to a grayscale image with equalized histogram
    """
    equalized_histogram_grayscale_image = pcv.hist_equalization(grayscale_image)
    return equalized_histogram_grayscale_image

def transform_histogram_equalized_grayscale_image_to_median_filtered_image(
        equalized_histogram_grayscale_image: numpy.ndarray
) -> numpy.ndarray:
    """
    Transforms histogram equalized grayscale image at path to a one
    which has a median filter applied to it
    """
    median_filtered_equalized_histogram_grayscale_image = pcv.median_blur(equalized_histogram_grayscale_image, 5)
    return median_filtered_equalized_histogram_grayscale_image

def transform_median_filtered_equalized_histogram_grayscale_image_to_otsu_segmented_image(
        median_filtered_equalized_histogram_grayscale_image: numpy.ndarray
) -> numpy.ndarray:
    """
    Transforms median filtered equalized histogram
    equalized grayscale image at path to otsu segmentation
    binary mask
    """
    otsu_segmentation_binary_mask = pcv.threshold.otsu(
        median_filtered_equalized_histogram_grayscale_image,
        object_type="dark"
    )
    return otsu_segmentation_binary_mask

def segment_grayscale_image_with_otsu_binary_mask(
        otsu_binary_mask: numpy.ndarray,
        grayscale_image: numpy.ndarray
) -> numpy.ndarray:
    """
    Otsu segments grayscale image using binary mask (background subtraction)
    """
    segmented_image = pcv.apply_mask(
        img=grayscale_image,
        mask=otsu_binary_mask,
        mask_color="black"
    )

    return segmented_image