import os
import numpy

from Utilities.util import (create_directory_if_needed, save_image, count_files_in_directory)
from project_configuration import Otsu_Segmentation_Binary_Mask_Image_Directory
from plantcv import plantcv as pcv
from Preprocessing.compute_all_training_data import compute_all_training_data
from plant_village_tomato_leaf_image_directories import plant_village_tomato_leaf_image_directories

def generate_otsu_segmentation_binary_masks():
    """
    Generate all otsu segmentation binary masks
    from median filtered histogram equalized
    grayscale image produced from the training data
    """
    training_data = compute_all_training_data()

    # Create all otsu segmented image directories
    create_all_otsu_segmentation_binary_leaf_image_mask_directories()

    for data in training_data:
        otsu_binary_mask = otsu_segmented_image(
            median_filtered_equalized_histogram_grayscale_image_path=data.median_filtered_equalized_grayscale_image_path
        )

        save_image(
            destination=data.otsu_binary_mask_image_path,
            img=otsu_binary_mask
        )

def otsu_segmented_image(
        median_filtered_equalized_histogram_grayscale_image_path: str
) -> numpy.ndarray:
    """
    Transforms median filtered equalized histogram
    equalized grayscale image at path to otsu segmentation
    binary mask
    """
    median_filtered_equalized_histogram_grayscale_image, _, _ = pcv.readimage(
        median_filtered_equalized_histogram_grayscale_image_path,
        mode='gray'
    )

    otsu_segmentation_binary_mask = pcv.threshold.otsu(median_filtered_equalized_histogram_grayscale_image, object_type="dark")
    return otsu_segmentation_binary_mask

def create_all_otsu_segmentation_binary_leaf_image_mask_directories():
    """
    Creates all directories for the otsu segmentation binary masks
    produced from the training data.
    """
    tomato_leaf_image_categories = plant_village_tomato_leaf_image_directories()

    create_directory_if_needed(Otsu_Segmentation_Binary_Mask_Image_Directory)

    for category in tomato_leaf_image_categories:
        tomato_leaf_image_category_directory = os.path.join(
            Otsu_Segmentation_Binary_Mask_Image_Directory,
            category
        )

        create_directory_if_needed(tomato_leaf_image_category_directory)

generate_otsu_segmentation_binary_masks()