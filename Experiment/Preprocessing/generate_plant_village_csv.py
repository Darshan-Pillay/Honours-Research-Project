import os
import cv2
import numpy
from plantcv import plantcv as pcv
from io import TextIOWrapper
from typing import List

from project_configuration import (
    Plant_Village_Directory,
    Gray_Scale_Image_Directory,
    Training_Data_File_Path
)

"""
Generates a csv file containing training data with path Training_Data_File_Path.
"""
def generate_training_data_csv():
    print("This may take a while...")
    with open(Training_Data_File_Path, mode='w') as csvfile:
        generate_training_data_from_all_tomato_leaf_images(csv_file_descriptor=csvfile)

"""
Generates csv training data in csvfile with file descriptor csv_file_descriptor
for each tomato leaf category in the plant village dataset
"""
def generate_training_data_from_all_tomato_leaf_images(csv_file_descriptor: TextIOWrapper):
    tomato_leaf_data_set_directories = plant_village_tomato_leaf_image_directories()
    for tomato_leaf_image_category in tomato_leaf_data_set_directories:
        generate_training_data_for_tomato_leaf_category(
            tomato_leaf_category=tomato_leaf_image_category,
            csv_file_descriptor=csv_file_descriptor
        )

"""
Returns a list of all the tomato leaf image directories in the plant village data set
"""
def plant_village_tomato_leaf_image_directories() -> List[str]:
    return list(filter(is_tomato_data_directory, os.listdir(Plant_Village_Directory)))

"""
Returns true if directory string represents a tomato leaf image directory
"""
def is_tomato_data_directory(directory: str) -> bool:
    if directory.startswith("Tomato"):
        return True

    return False

"Generates csv training data for every tomato leaf plant category in the plant village data set"
def generate_training_data_for_tomato_leaf_category(tomato_leaf_category: str, csv_file_descriptor: TextIOWrapper):
    tomato_leaf_images = os.listdir(os.path.join(Plant_Village_Directory, tomato_leaf_category))
    for tomato_leaf_image in tomato_leaf_images:
        rgb_leaf_image_path = os.path.join(
            Plant_Village_Directory,
            tomato_leaf_category,
            tomato_leaf_image
        )

        grayscale_leaf_image_file_name = generate_grayscale_leaf_image(
            rgb_leaf_image_path=rgb_leaf_image_path,
            grayscale_leaf_image_filename=tomato_leaf_image,
            tomato_leaf_image_category=tomato_leaf_category
        )

        write_training_datum_to_csvfile(
            csv_file_descriptor=csv_file_descriptor,
            tomato_leaf_image_category=tomato_leaf_category,
            rgb_leaf_image_path=rgb_leaf_image_path,
            grayscale_leaf_image_filename=grayscale_leaf_image_file_name
        )

"""
Writes a string of the form 'tomato_leaf_image_category, rgb_leaf_image_path, is_image_for_diseased_plant
grayscale_image_filename' to the csv file with handle csv_file_descriptor.

For example, if tomato_leaf_image_category is Tomato_Late_Blight, rgb_leaf_image_path is /example.jpg,
grayscale_leaf_image_file_name is /grayscale/example.jpg, then the line
Tomato_Late_Blight,/example.jpg,/grayscale/example.jpg,true is written to csv file.
"""
def write_training_datum_to_csvfile(
        csv_file_descriptor: TextIOWrapper,
        tomato_leaf_image_category: str,
        rgb_leaf_image_path: str,
        grayscale_leaf_image_filename: str
):
    is_image_for_diseased_plant = False if "healthy" in tomato_leaf_image_category else True

    training_datum = [
        tomato_leaf_image_category,
        rgb_leaf_image_path,
        grayscale_leaf_image_filename,
        str(is_image_for_diseased_plant)
    ]

    csv_file_descriptor.write(",".join(training_datum) + "\n")

"""
Creates a directory at path if it does not exist
"""
def create_directory_if_needed(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

"""
Writes img to disk at path destination
"""
def save_image(destination: str, img: numpy.ndarray):
    cv2.imwrite(destination, img)

"""
Transforms RGB image at rgb_image_path to a grayscale image
"""
def transform_rgb_image_to_grayscale(rgb_image_path: str) -> numpy.ndarray:
    rgb_image, _, _ = pcv.readimage(rgb_image_path, mode="rgb")
    grayscale_image = pcv.rgb2gray(rgb_img=rgb_image)
    return grayscale_image

"""
Converts an rgb image at path rgb_leaf_image_path to grayscale, then saves it
at ROOT_Directory/Data Set/GrayScale/tomato_leaf_image_category/grayscale_leaf_image_filename,
and returns the path of the saved grayscale image.

For example, given ../PlantVillage/Tomato_Late_Blight/example_image.JPG as rgb_leaf_image_path, 
grayscale_leaf_image_filename as example_image.JPG, and tomato_leaf_image_category as Tomato_Late_Blight,
then the image ../PlantVillage/Tomato_Late_Blight/example_image.JPG is transformed to grayscale, 
and then this grayscale is saved to disk at 
ROOT_Directory/Data Set/GrayScale/Tomato_Late_Blight/example_image.JPG, and this path is returned to a 
caller of this function
"""
def generate_grayscale_leaf_image(
    rgb_leaf_image_path: str,
    grayscale_leaf_image_filename: str,
    tomato_leaf_image_category: str,
) -> str:
    # Create directory if needed
    create_directory_if_needed(path=Gray_Scale_Image_Directory)
    grayscale_leaf_image_directory = os.path.join(Gray_Scale_Image_Directory, tomato_leaf_image_category)
    create_directory_if_needed(path=grayscale_leaf_image_directory)

    # Transform RGB leaf image to grayscale
    grayscale_leaf_img = transform_rgb_image_to_grayscale(rgb_image_path=rgb_leaf_image_path)

    # Save grayscale image to disk
    grayscale_leaf_image_filename = os.path.join(grayscale_leaf_image_directory, grayscale_leaf_image_filename)
    save_image(destination=grayscale_leaf_image_filename, img=grayscale_leaf_img)

    # Return path of saved grayscale leaf image
    return grayscale_leaf_image_filename

"""
Applies histogram equalization to a gray scale image
"""
def equalize_grayscale_image(grayscale_img: numpy.ndarray) -> numpy.ndarray:
    return pcv.hist_equalization(grayscale_img)

"""
Apply median filter to grayscale image
"""
def median_filter_grayscale_image(grayscale_img: numpy.ndarray) -> numpy.ndarray:
    return pcv.median_blur(grayscale_img, 5)

generate_training_data_csv()