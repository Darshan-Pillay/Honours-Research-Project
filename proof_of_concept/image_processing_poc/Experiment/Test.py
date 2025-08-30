# Matplot Lib
import matplotlib
import matplotlib.pyplot as plt

# Plant CV
from plantcv import plantcv as pcv

# SVM Machine Learning
from skimage.feature import graycomatrix, graycoprops
from sklearn import svm

# Miscellenous
import os
import argparse
import numpy as py
from pathlib import Path

# Set-up Plant CV Global Parameters
class PlantCV_Configuration:
    def __init__(self):
        self.debug = "plot"
        self.writeimg = False
        self.result = ""
        self.outdir = "."

plant_cv_config = PlantCV_Configuration()
pcv.params.debug_outdir = plant_cv_config.outdir
pcv.params.debug = plant_cv_config.debug

# Get Path Of Data Set
current_directory = Path().cwd()
image_data_set_directory = current_directory / 'Plant_leave_diseases_dataset_without_augmentation'
image_data_set_directory.resolve()

poc_200_image_directory = image_data_set_directory / 'Apple___Apple_scab'
poc_200_image_directory.resolve()

# Read the first 200 images
first_200_images = [str(image_path) for image_path in list(poc_200_image_directory.iterdir())[:1]]

# Load all the images
opened_images = list(map(lambda image_path: pcv.readimage(image_path, mode="rgb")[0], first_200_images))

# Update params
pcv.params.text_size = 50
pcv.params.text_thickness = 15

# Grat Scale converted image
gray_scaled_images = list(map(lambda image: pcv.rgb2gray(rgb_img=image), opened_images))

# Histogram Equalized gray scale images
equalized_gray_scaled_images = list(map(lambda gray_scale_image: pcv.hist_equalization(gray_scale_image), gray_scaled_images))

# Median Filtering
equalized_median_filtered_images =  list(map(lambda equalized_image: pcv.median_blur(equalized_image, 5), equalized_gray_scaled_images))

# Segmentation With Otsu Method
segmented_images_with_otsu = list(map(lambda median_filtered_image: pcv.threshold.otsu(median_filtered_image, object_type="light"), equalized_median_filtered_images))

# Close Holes (Morphological Operation)
filled_images = list(map(lambda segmented_image_with_otsu: pcv.fill_holes(bin_img=segmented_image_with_otsu), segmented_images_with_otsu))