import os

# Project Root
ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

Data_Set_Directory = ROOT_DIRECTORY + "/DataSet"

Plant_Village_Directory = Data_Set_Directory + "/PlantVillage"

Gray_Scale_Image_Directory = Data_Set_Directory + "/GrayScale"

Equalized_Gray_Scale_Image_Directory = Data_Set_Directory + "/EqualizedGrayScale"

Median_Filtered_Equalized_Gray_Scale_Image_Directory = Data_Set_Directory + "/MedianFilteredEqualizedGrayScale"

Otsu_Segmentation_Binary_Mask_Image_Directory = Data_Set_Directory + "/OtsuBinaryMask"

Segmented_Image_Directory = Data_Set_Directory + "/OtsuSegmentedImages"

Feature_Vector_Directory = Data_Set_Directory + "/FeatureVectors"

Training_Data_File_Directory = Data_Set_Directory + "/TrainingData/"
Training_Data_File_Path = Training_Data_File_Directory + "/training_data.csv"