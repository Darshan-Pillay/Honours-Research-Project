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
import random
import numpy as py
from pathlib import Path

# Set-up Plant CV Global Parameters
class PlantCV_Configuration:
    def __init__(self):
        self.debug = ""
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
first_200_images = [str(image_path) for image_path in list(poc_200_image_directory.iterdir())[:200]]

# Load all the images
opened_images = list(map(lambda image_path: pcv.readimage(image_path, mode="rgb")[0], first_200_images))

# Update params
pcv.params.text_size = 50
pcv.params.text_thickness = 15

# Gray Scale converted image
gray_scaled_images = list(map(lambda image: pcv.rgb2gray(rgb_img=image), opened_images))

# Histogram Equalized gray scale images
equalized_gray_scaled_images = list(map(lambda gray_scale_image: pcv.hist_equalization(gray_scale_image), gray_scaled_images))

# Median Filtering
equalized_median_filtered_images =  list(map(lambda equalized_image: pcv.median_blur(equalized_image, 5), equalized_gray_scaled_images))

# Segmentation With Otsu Method
segmented_images_with_otsu = list(map(lambda median_filtered_image: pcv.threshold.otsu(median_filtered_image, object_type="light"), equalized_median_filtered_images))

# Apply Mask From Otsu Method
background_removed_using_mask = list(map(lambda gray_scale_segmented_image_pair: pcv.apply_mask(img=gray_scale_segmented_image_pair[0], mask=gray_scale_segmented_image_pair[1], mask_color="black"), zip(gray_scaled_images, segmented_images_with_otsu)))

# Calculate gray-scale co-occurence matrics
glcms = list(map(lambda feature_image: graycomatrix(feature_image, distances=[1], angles=[45], levels=256, symmetric=True, normed=True), background_removed_using_mask))

# Obtain Training Data for SVM
def calculate_feature_vector(glcm: py.ndarray) -> tuple[py.ndarray, int]:
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop="energy")
    homogeneity = graycoprops(glcm, prop="homogeneity")
    correlation = graycoprops(glcm, prop="correlation")
    dissimilarity = graycoprops(glcm, prop="dissimilarity")
    ASM = graycoprops(glcm, prop="ASM")
    mean = graycoprops(glcm, prop="mean")
    variance = graycoprops(glcm, prop="variance")
    std = graycoprops(glcm, prop="std")
    entropy = graycoprops(glcm, prop="entropy")

    feature_vector = py.array([contrast[0][0], energy[0][0], homogeneity[0][0], correlation[0][0], dissimilarity[0][0], ASM[0][0], mean[0][0], variance[0][0], std[0][0], entropy[0][0]])
    category_label = random.randint(1, 2)
    return feature_vector, category_label

training_data = list(map(lambda glcm: calculate_feature_vector(glcm), glcms))

# SVM Training
training_samples = [training_sample[0] for training_sample in training_data]
training_sample_labels = [training_sample[1] for training_sample in training_data]

classifier = svm.SVC()
classifier.fit(training_samples,training_sample_labels)

# Classifier Predictions
prediction = classifier.predict(py.array([[0,1,2,3,4,5,6,7,8,9]]))
print(prediction)

# split the training data set, train the SVM, and then predict various measures

# 1. Need to understand the support vector machine properly
# 2. What is kernal, degree etc
# 3. Need to know how to split the data_set, and how to cross-validate