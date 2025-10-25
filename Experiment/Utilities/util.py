import os

"""
Creates a directory at path if it does not exist
"""
def create_directory_if_needed(path: str):
    if not os.path.exists(path):
        os.mkdir(path)