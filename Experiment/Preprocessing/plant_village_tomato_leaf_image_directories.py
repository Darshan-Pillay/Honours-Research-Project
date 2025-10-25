import os

from typing import List

from project_configuration import Plant_Village_Directory

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