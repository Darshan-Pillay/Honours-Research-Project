import os
import cv2
import numpy

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