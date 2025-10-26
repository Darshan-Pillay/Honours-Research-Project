from Classes import CsvDataSetItem
from csv import DictWriter
from Preprocessing.Utilities import compute_all_training_data_info, create_directory_if_needed

from ProjectConfiguration import (
    Training_Data_File_Directory,
    Training_Data_Csv_File_Path
)

def generate_csv_file():
    """
    Generates csv training data file
    """
    training_data = compute_all_training_data_info()
    write_training_data_to_csvfile(training_data)

def write_training_data_to_csvfile(data: list[CsvDataSetItem]):
    """
    Writes each CsvDataSetItem value in list to a csv.
    """
    create_directory_if_needed(Training_Data_File_Directory)
    with open(Training_Data_Csv_File_Path, mode="w", newline='') as csvfile:

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