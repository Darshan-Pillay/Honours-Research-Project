import os
import cv2
import numpy
import sklearn

from joblib import dump, load

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