import sklearn
import os
import cv2
import numpy

from joblib import dump, load
from Preprocessing.Classes import CsvDataSetItem
from typing import List

from ProjectConfiguration import (
    Plant_Village_Directory,
    Gray_Scale_Image_Directory,
    Equalized_Gray_Scale_Image_Directory,
    Median_Filtered_Equalized_Gray_Scale_Image_Directory,
    Otsu_Segmentation_Binary_Mask_Image_Directory,
    Segmented_Image_Directory,
    Feature_Vector_Directory
)

def compute_all_training_data_info() -> list[CsvDataSetItem]:
    """
    :return: A list of all training csv data items
    """
    data_items: List[CsvDataSetItem] = []

    tomato_leaf_data_set_directories = plant_village_tomato_leaf_image_directories()
    for tomato_leaf_image_category in tomato_leaf_data_set_directories:

        leaf_images_for_category = os.listdir(
            os.path.join(
                Plant_Village_Directory,
                tomato_leaf_image_category
            )
        )

        for tomato_leaf_image_name in leaf_images_for_category:
            data_item = construct_training_data_item(
                tomato_leaf_image_category,
                tomato_leaf_image_name
            )
            data_items.append(data_item)

    return data_items

def construct_training_data_item(
        tomato_leaf_image_category: str,
        tomato_leaf_image_name: str
) -> CsvDataSetItem:
    """
    Computes a data item in the form of CsvDataSetItem

    :param tomato_leaf_image_category: This represents the category that this image falls under. For example
    in the PlantVillage data set a health tomato may be under the directory PlantVillage/Tomato_healthy.
    Here the leaf_image category is Tomato_healthy

    :param tomato_leaf_image_name: This represents the file name of the leaf image. For example
    in the PlantVillage data set a health tomato may be under the directory PlantVillage/Tomato_healthy/1810801823.jpg.
    Here the tomato_leaf_image_name is 1810801823.jpg

    :return: CsvDataSetItem
    """

    original_rgb_leaf_image_path = os.path.join(
        Plant_Village_Directory,
        tomato_leaf_image_category,
        tomato_leaf_image_name
    )

    grayscale_leaf_image_path = os.path.join(
        Gray_Scale_Image_Directory,
        tomato_leaf_image_category,
        tomato_leaf_image_name
    )

    equalized_grayscale_image_path = os.path.join(
        Equalized_Gray_Scale_Image_Directory,
        tomato_leaf_image_category,
        tomato_leaf_image_name
    )

    median_filtered_equalized_grayscale_image_path = os.path.join(
        Median_Filtered_Equalized_Gray_Scale_Image_Directory,
        tomato_leaf_image_category,
        tomato_leaf_image_name
    )

    otsu_binary_mask_image_path = os.path.join(
        Otsu_Segmentation_Binary_Mask_Image_Directory,
        tomato_leaf_image_category,
        tomato_leaf_image_name
    )

    otsu_segmented_image_path = os.path.join(
        Segmented_Image_Directory,
        tomato_leaf_image_category,
        tomato_leaf_image_name
    )

    feature_vector_path = os.path.join(
        Feature_Vector_Directory,
        tomato_leaf_image_category,
        tomato_leaf_image_name + ".npy" # we serialize our feature vectors to disk using numpy
    )

    is_diseased_leaf_image = False if "healthy" in tomato_leaf_image_category else True

    data_item = CsvDataSetItem(
        tomato_leaf_image_category,
        original_rgb_leaf_image_path,
        grayscale_leaf_image_path,
        equalized_grayscale_image_path,
        median_filtered_equalized_grayscale_image_path,
        otsu_binary_mask_image_path,
        otsu_segmented_image_path,
        feature_vector_path,
        is_diseased_leaf_image
    )

    return data_item

def plant_village_tomato_leaf_image_directories() -> List[str]:
    """
    Returns a list of all the tomato leaf image directories in the plant village data set
    """
    return list(filter(is_tomato_data_directory, os.listdir(Plant_Village_Directory)))

def is_tomato_data_directory(directory: str) -> bool:
    """
    Returns true if directory string represents a tomato leaf image directory
    """
    if directory.startswith("Tomato"):
        return True

    return False

def create_directory_if_needed(path: str):
    """
    Creates a directory at path if it does not exist
    """
    if not os.path.exists(path):
        os.mkdir(path)

def save_image(destination: str, img: numpy.ndarray):
    """
    Writes img to disk at path destination
    """
    cv2.imwrite(destination, img)

def count_files_in_directory(directory_path: str) -> int:
    """
    Counts all files in a directory and its subdirectories. We used this function
    to verify our preprocessing steps generate the expected number of images. For
    example the database we used for our research contains 16011 tomato leaf images,
    so we can use this function to verify that each preprocessing step saves to disk
    the correct number of images.
    """
    total_files = 0
    for root, _, files in os.walk(directory_path):
        total_files += len(files)
    return total_files

def save_model_to_disk(
        model: sklearn.svm._classes,
        destination_path: str
):
    """
    Saves model to disk
    """
    dump(model, destination_path)

def load_model_from_disk(model_path: str) -> sklearn.svm._classes:
    """
    Loads model from path (assuming model saved using joblib)
    """
    return load(model_path)