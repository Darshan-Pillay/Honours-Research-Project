import pickle
import numpy

from typing import List
from sklearn.model_selection import train_test_split
from Preprocessing.Classes import TrainingItem
from Preprocessing.Utilities import compute_all_training_data_info, create_directory_if_needed
from pathlib import Path

from ProjectConfiguration import (
    Training_Data_File_Directory,
    Training_Data_File_Path,
    Test_Data_File_Directory,
    Test_Data_File_Path
)

def generate_training_data_and_test_data():
    """
    Splits all data into training set, and test set,
    and saves this split to disk. This allows for
    better reproducablity in the future should
    anyone else wish to verify, extend, refute
    the work I produced here.
    """
    all_feature_ids, all_feature_vectors, all_instance_labels = load_training_data_from_disk()

    # Training and test split
    (training_features,
     test_features,
     training_instance_labels,
     test_instance_labels,
     training_feature_ids,
     test_feature_ids
     ) = train_test_split(
        all_feature_vectors,
        all_instance_labels,
        all_feature_ids,
        test_size=0.1,
        train_size=0.9,
        random_state=42,
        shuffle=True,
        stratify=all_instance_labels
    )

    # serialize training data to disk to allow for better reproducablity
    training_items: List[TrainingItem] = []
    for training_item_id, feature, label in zip(training_feature_ids, training_features, training_instance_labels):
        item = TrainingItem(
            id=training_item_id,
            texture_feature=feature,
            is_item_a_diseased_leaf_image=label
        )
        training_items.append(item)

    create_directory_if_needed(Training_Data_File_Directory)
    with open(Training_Data_File_Path, "wb") as training_data_file:
        pickle.dump(training_items, training_data_file)

    # serialize test data to disk to allow for better reproducablity
    test_items: List[TrainingItem] = []
    for test_item_id, feature, label in zip(test_feature_ids, test_features, test_instance_labels):
        item = TrainingItem(
            id=test_item_id,
            texture_feature=feature,
            is_item_a_diseased_leaf_image=label
        )
        test_items.append(item)

    create_directory_if_needed(Test_Data_File_Directory)
    with open(Test_Data_File_Path, mode="wb") as test_data_file:
        pickle.dump(test_items, test_data_file)

def load_training_data_from_disk() -> (
        List[str],
        List[numpy.ndarray],
        List[bool]
):
    """
    Returns all feature vectors for our training data,
    and their corresponding class labels, as well as the
    image ids for which this data was obtained from.

    Example: if the feature vector was obtained from /path/img_12223.jpg,
    and the feature vector is [1,2,3], and its label is 1, then
    (img_12223.jpg, [1,2,3], 1) will be a 3-tuple returned.
    """
    training_data_info = compute_all_training_data_info()

    feature_ids: List[str] = []
    feature_vectors: List[numpy.ndarray] = []
    feature_vector_labels: List[bool] = []

    for datum in training_data_info:
        feature_ids.append(Path(datum.original_rgb_image_path).name)
        feature_vectors.append(datum.feature_vector())
        feature_vector_labels.append(datum.is_part_of_target_category)

    return feature_ids, feature_vectors, feature_vector_labels

def load_training_data_from_pickle_object() -> (List[numpy.ndarray], List[bool]):
    """
    Loads training data from pickle training data object
    """
    # load training data from pickle object
    features: List[numpy.ndarray] = []
    feature_label: List[bool] = []
    with open(Training_Data_File_Path, mode="rb") as training_data_file:
        training_data: List[TrainingItem] = pickle.load(training_data_file)
        for data in training_data:
            features.append(data.texture_feature)
            feature_label.append(data.is_item_a_diseased_leaf_image)
    return features, feature_label

def load_test_data_from_pickle_object() -> (List[numpy.ndarray], List[bool]):
    # load training data from pickle object
    features: List[numpy.ndarray] = []
    feature_label: List[bool] = []
    with open(Test_Data_File_Path, mode="rb") as test_data_file:
        test_data: List[TrainingItem] = pickle.load(test_data_file)
        for data in test_data:
            features.append(data.texture_feature)
            feature_label.append(data.is_item_a_diseased_leaf_image)
    return features, feature_label