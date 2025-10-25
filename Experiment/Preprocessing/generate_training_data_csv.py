from CsvDataSetItem import CsvDataSetItem
from Utilities.util import create_directory_if_needed
from csv import DictWriter
from Preprocessing.compute_all_training_data import compute_all_training_data

from project_configuration import (
    Training_Data_File_Directory,
    Training_Data_File_Path
)

def generate_csv_file():
    """
    Generates csv training data file
    """
    training_data = compute_all_training_data()
    write_training_data_to_csvfile(training_data)

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

generate_csv_file()