import os
import numpy

from Utilities.util import (create_directory_if_needed, save_image, count_files_in_directory)
from project_configuration import Segmented_Image_Directory
from plantcv import plantcv as pcv
from Preprocessing.compute_all_training_data import compute_all_training_data
from plant_village_tomato_leaf_image_directories import plant_village_tomato_leaf_image_directories

def generate_otsu_segmented_images():
    """
    Generate all otsu segmented images from training data
    """
    training_data = compute_all_training_data()

    # Create all otsu segmented images directories
    create_all_otsu_segmented_image_directories()

    for data in training_data:
        segmented_image = otsu_segmented_image(
            otsu_binary_mask=data.otsu_binary_mask_image_path,
            grayscale_image=data.grayscale_image_path
        )

        save_image(
            destination=data.otsu_segmented_image_path,
            img=segmented_image
        )

def otsu_segmented_image(
        otsu_binary_mask: str,
        grayscale_image: str
) -> numpy.ndarray:
    """
    Otsu segments grayscale image using binary mask (background subtraction)
    """
    grayscale_image, _, _ = pcv.readimage(grayscale_image, mode="gray")
    otsu_binary_mask, _, _ = pcv.readimage(otsu_binary_mask, mode='gray')

    segmented_image = pcv.apply_mask(
        img=grayscale_image,
        mask=otsu_binary_mask,
        mask_color="black"
    )

    return segmented_image

def create_all_otsu_segmented_image_directories():
    """
    Creates all directories for the otsu segmented images
    produced from the training data.
    """
    tomato_leaf_image_categories = plant_village_tomato_leaf_image_directories()

    create_directory_if_needed(Segmented_Image_Directory)

    for category in tomato_leaf_image_categories:
        tomato_leaf_image_category_directory = os.path.join(
            Segmented_Image_Directory,
            category
        )

        create_directory_if_needed(tomato_leaf_image_category_directory)

generate_otsu_segmented_images()