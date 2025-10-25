import os
import numpy

from Utilities.util import (create_directory_if_needed, save_image, count_files_in_directory)
from project_configuration import Median_Filtered_Equalized_Gray_Scale_Image_Directory
from plantcv import plantcv as pcv
from Preprocessing.compute_all_training_data import compute_all_training_data
from plant_village_tomato_leaf_image_directories import plant_village_tomato_leaf_image_directories

def generate_median_filtered_histogram_equalized_grayscale_leaf_images():
    """
    Generate all median filtered equalized histogram grayscale
    leaf images from the training data,  and then saves them to disk
    """
    training_data = compute_all_training_data()

    # Create all median filtered equalized histogram grayscale image directories
    create_all_median_filtered_equalized_grayscale_leaf_image_directories()

    for data in training_data:
        median_filtered_histogram_equalized_grayscale_leaf_image = median_filtered_histogram_equalized_grayscale_image(
            equalized_histogram_grayscale_image_path=data.equalized_grayscale_image_path
        )

        save_image(
            destination=data.median_filtered_equalized_grayscale_image_path,
            img=median_filtered_histogram_equalized_grayscale_leaf_image
        )

def median_filtered_histogram_equalized_grayscale_image(
        equalized_histogram_grayscale_image_path: str
) -> numpy.ndarray:
    """
    Transforms histogram equalized grayscale image at path to a one
    which has a median filter applied to it
    """
    equalized_histogram_grayscale_image, _, _ = pcv.readimage(equalized_histogram_grayscale_image_path, mode='gray')
    median_filtered_equalized_histogram_grayscale_image = pcv.median_blur(equalized_histogram_grayscale_image, 5)
    return median_filtered_equalized_histogram_grayscale_image

def create_all_median_filtered_equalized_grayscale_leaf_image_directories():
    """
    Creates all directories for the median filtered equalized gray scale images produced
    from the training data
    """
    tomato_leaf_image_categories = plant_village_tomato_leaf_image_directories()

    create_directory_if_needed(Median_Filtered_Equalized_Gray_Scale_Image_Directory)

    for category in tomato_leaf_image_categories:
        tomato_leaf_image_category_directory = os.path.join(
            Median_Filtered_Equalized_Gray_Scale_Image_Directory,
            category
        )

        create_directory_if_needed(tomato_leaf_image_category_directory)

generate_median_filtered_histogram_equalized_grayscale_leaf_images()