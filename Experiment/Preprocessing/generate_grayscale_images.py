import os
import numpy

from Utilities.util import (create_directory_if_needed, save_image)
from project_configuration import Gray_Scale_Image_Directory
from plantcv import plantcv as pcv
from Preprocessing.compute_all_training_data import compute_all_training_data
from plant_village_tomato_leaf_image_directories import plant_village_tomato_leaf_image_directories

def generate_grayscale_leaf_images():
    """
    Generate all grayscale leaf images from the training data,
    and then saves them to disk
    """
    training_data = compute_all_training_data()

    # Create all grayscale image directories
    create_all_grayscale_leaf_image_directories()

    for data in training_data:
        grayscale_leaf_image = transform_rgb_image_to_grayscale(
            rgb_image_path=data.original_rgb_image_path
        )

        save_image(
            destination=data.grayscale_image_path,
            img=grayscale_leaf_image
        )

def transform_rgb_image_to_grayscale(rgb_image_path: str) -> numpy.ndarray:
    """
    Transforms RGB image at rgb_image_path to a grayscale image
    """
    rgb_image, _, _ = pcv.readimage(rgb_image_path, mode="rgb")
    grayscale_image = pcv.rgb2gray(rgb_img=rgb_image)
    return grayscale_image

def create_all_grayscale_leaf_image_directories():
    """
    Creates all directories for the gray scale images produced
    from the training data
    """
    tomato_leaf_image_categories = plant_village_tomato_leaf_image_directories()

    create_directory_if_needed(Gray_Scale_Image_Directory)

    for category in tomato_leaf_image_categories:
        tomato_leaf_image_category_directory = os.path.join(
            Gray_Scale_Image_Directory,
            category
        )

        create_directory_if_needed(tomato_leaf_image_category_directory)