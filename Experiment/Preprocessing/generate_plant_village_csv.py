import os
from project_configuration import Plant_Village_Directory, all_plant_village_tomato_directories

"""
Goes through each tomato directory in the PlantVillage directory, and creates our experiment csv.
The format of the csv for each line is: image_category, image_path

For example, if the plant village directory contains a folder Tomato_Healthy which has an image named 'example.jpg',
then the csv will have a line 'Tomato_Healthy, example.jpg'
"""
def generate_plant_village_csv():
    with open('../plant_village.csv', mode='w') as csvfile:
        data_set_directories = all_plant_village_tomato_directories()

        for directory in data_set_directories:
            for directory_file in os.listdir(Plant_Village_Directory + "/" + directory):
                csvfile.write(directory + "," + directory_file + "\n")