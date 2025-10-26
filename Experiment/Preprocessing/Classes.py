from typing import List
import numpy
from Preprocessing.FeatureExtraction import load_texture_vector_from_disk

class TrainingItem:
    """
    Class representing a training item pair
    For example if the pair is (feature_vector, True)
    Then this item represents a feature vector for an item
    that is part of the category of interest

    Note: The id property represent the path-suffix of a training data image
    This can be useful to see which training data was used and what test
    data is used
    """
    def __init__(
            self,
            id: str,
            texture_feature: numpy.ndarray,
            is_item_a_diseased_leaf_image
    ):
        self.id = id
        self.texture_feature = texture_feature
        self.is_item_a_diseased_leaf_image = is_item_a_diseased_leaf_image

class CsvDataSetItem:
    """
    Represents one training vector for this research. Each member value of this class
    will form one value in a single row of the training data CSv
    """

    def __init__(
            self,
            item_label: str,
            original_rgb_image_path: str,
            grayscale_image_path: str,
            equalized_grayscale_image_path: str,
            median_filtered_equalized_grayscale_image_path: str,
            otsu_binary_mask_image_path: str,
            otsu_segmented_image_path: str,
            feature_vector_path: str,
            is_part_of_target_category: bool
    ):
        self.item_label = item_label
        self.original_rgb_image_path = original_rgb_image_path
        self.grayscale_image_path = grayscale_image_path
        self.equalized_grayscale_image_path = equalized_grayscale_image_path
        self.median_filtered_equalized_grayscale_image_path = median_filtered_equalized_grayscale_image_path
        self.otsu_binary_mask_image_path = otsu_binary_mask_image_path
        self.otsu_segmented_image_path = otsu_segmented_image_path
        self.feature_vector_path = feature_vector_path
        self.is_part_of_target_category = is_part_of_target_category

    @staticmethod
    def attributes() -> List[str]:
        """
        :return: A list of strings representing the headers or attribute names
        of each data value in this CsvData Item. For example, in a csv if all
        rows have columns (Name, Surname, Age) then the return value of this function
        is ["Name", "Surname", "Age"]
        """

        return [attr for attr in CsvDataSetItem.__init__.__code__.co_varnames if attr != 'self']

    def to_row_value(self) -> dict:
        """
        :return: A vector of each value in this instance as a dictionary
        """

        row_value: dict = self.__dict__
        return row_value

    def feature_vector(self) -> numpy.ndarray:
        """
        Loads and returns feature vector for this instance.
        The feature vector is at path on disk feature_vector_path
        """
        feature_vector_from_disk: numpy.ndarray = load_texture_vector_from_disk(
            feature_vector_path=self.feature_vector_path
        )

        return feature_vector_from_disk