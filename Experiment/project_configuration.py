import os
from typing import List

# Project Root
ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Data Set Related Paths
Plant_Village_Directory = ROOT_DIRECTORY + "/Data Set/PlantVillage"

def all_plant_village_tomato_directories() -> List[str]:
    return list(filter(is_tomato_data_directory, os.listdir(Plant_Village_Directory)))

def is_tomato_data_directory(directory: str) -> bool:
    if directory.startswith("Tomato"):
        return True

    return False