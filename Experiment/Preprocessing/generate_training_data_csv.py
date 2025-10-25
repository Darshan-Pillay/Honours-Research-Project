import csv
import os
from time import sleep

from typing import List
from CsvDataSetItem import CsvDataSetItem
from Utilities.util import create_directory_if_needed
from csv import DictWriter

from project_configuration import (
    Plant_Village_Directory,
    Gray_Scale_Image_Directory,
    Equalized_Gray_Scale_Image_Directory,
    Median_Filtered_Equalized_Gray_Scale_Image_Directory,
    Otsu_Segmentation_Binary_Mask_Image_Directory,
    Segmented_Image_Directory,
    Feature_Vector_Directory,
    Training_Data_File_Directory,
    Training_Data_File_Path
)

def generate_csv_file():
    """
    Generates csv training data file
    """
    training_data = compute_all_training_data()
    write_training_data_to_csvfile(training_data)

def compute_all_training_data() -> list[CsvDataSetItem]:
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

def write_training_data_to_csvfile(data: list[CsvDataSetItem]):
    """
    Writes each CsvDataSetItem value in list to a csv.
    """
    create_directory_if_needed(Training_Data_File_Directory)
    with open(Training_Data_File_Path, mode="w", newline='') as csvfile:

        writer = DictWriter(
            csvfile, delimiter=',',
            lineterminator='\n',
            fieldnames=CsvDataSetItem.attributes()
        )

        # Writes the header names/columns names to csv
        writer.writeheader()

        # Write each data to csv as a row
        for datum in data:
            writer.writerow(datum.to_row_value())

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

generate_csv_file()