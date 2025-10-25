import numpy

from skimage.feature.texture import (
    modified_graycomatrix,
    graycoprops
)


def compute_glcm_metrics(
        grayscale_img: numpy.ndarray
) -> numpy.ndarray:
    """
    Takes in a grayscale_img and computes a texture feature vector based on GLCM metrics.
    """

    # Calculate GLCM
    distances = [1] # we can increase this, but lose textural resolution
    angles = [0, 45, 90, 135]
    levels = 256 # assume 8bit image

    glcm = modified_graycomatrix(
        grayscale_img.astype(numpy.int8),
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
        sentinel_value=-1
    )

    # Calculate GLCM Metrics
    contrast: numpy.ndarray = graycoprops(glcm, prop="contrast")
    energy: numpy.ndarray = graycoprops(glcm, prop="energy")
    homogeneity: numpy.ndarray = graycoprops(glcm, prop="homogeneity")
    correlation: numpy.ndarray = graycoprops(glcm, prop="correlation")
    dissimilarity: numpy.ndarray = graycoprops(glcm, prop="dissimilarity")
    ASM: numpy.ndarray = graycoprops(glcm, prop="ASM")
    mean: numpy.ndarray = graycoprops(glcm, prop="mean")
    variance: numpy.ndarray = graycoprops(glcm, prop="variance")
    std: numpy.ndarray = graycoprops(glcm, prop="std")
    entropy: numpy.ndarray = graycoprops(glcm, prop="entropy")

    # Create Texture Feature Vector
    texture_feature_vector = numpy.array([
        contrast,
        energy,
        homogeneity,
        correlation,
        dissimilarity,
        ASM,
        mean,
        variance,
        std,
        entropy
    ]).flatten()

    return texture_feature_vector

"""
Saves texture vector to disk
"""
def save_texture_vector_to_disk(
        feature_vector: numpy.ndarray,
        destination_path: str
):
    numpy.save(destination_path, feature_vector, allow_pickle=False)

"""
Loads texture vector from disk
"""
def load_texture_vector_from_disk(
        feature_vector_path: str
) -> numpy.ndarray:
    return numpy.load(feature_vector_path, allow_pickle=False)