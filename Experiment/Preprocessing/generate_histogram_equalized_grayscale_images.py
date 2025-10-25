import os
import numpy

from Utilities.util import (create_directory_if_needed, save_image)
from project_configuration import Equalized_Gray_Scale_Image_Directory
from plantcv import plantcv as pcv
from Preprocessing.compute_all_training_data import compute_all_training_data
from plant_village_tomato_leaf_image_directories import plant_village_tomato_leaf_image_directories

def generate_histogram_equalized_grayscale_leaf_images():
    """
    Generate all grayscale leaf images from the training data,
    and then saves them to disk
    """
    training_data = compute_all_training_data()

    # Create all grayscale image directories
    create_all_equalized_grayscale_leaf_image_directories()

    for data in training_data:
        grayscale_leaf_image = histogram_equalized_grayscale_image(
            grayscale_image_path=data.grayscale_image_path
        )

        save_image(
            destination=data.equalized_grayscale_image_path,
            img=grayscale_leaf_image
        )

def histogram_equalized_grayscale_image(grayscale_image_path: str) -> numpy.ndarray:
    """
    Transforms grayscale image at path to a grayscale image with equalized histogram
    """
    grayscale_image, _, _ = pcv.readimage(grayscale_image_path, mode='gray')
    equalized_histogram_grayscale_image = pcv.hist_equalization(grayscale_image)
    return equalized_histogram_grayscale_image

def create_all_equalized_grayscale_leaf_image_directories():
    """
    Creates all directories for the equalized gray scale images produced
    from the training data
    """
    tomato_leaf_image_categories = plant_village_tomato_leaf_image_directories()

    create_directory_if_needed(Equalized_Gray_Scale_Image_Directory)

    for category in tomato_leaf_image_categories:
        tomato_leaf_image_category_directory = os.path.join(
            Equalized_Gray_Scale_Image_Directory,
            category
        )

        create_directory_if_needed(tomato_leaf_image_category_directory)

generate_histogram_equalized_grayscale_leaf_images()