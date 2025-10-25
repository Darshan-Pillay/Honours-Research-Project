from typing import List

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