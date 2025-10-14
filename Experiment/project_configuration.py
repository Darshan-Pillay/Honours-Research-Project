import os

# Project Root
ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Data Set Related Paths
Data_Set_Directory = ROOT_DIRECTORY + "/DataSet"
Plant_Village_Directory = Data_Set_Directory + "/PlantVillage"
Gray_Scale_Image_Directory = Data_Set_Directory + "/GrayScale"

# Training Data Related Paths
Training_Data_File_Path = ROOT_DIRECTORY + "/TrainingData/training_data.csv"