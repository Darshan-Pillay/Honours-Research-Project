import sklearn
import numpy as np

def strict_equal_svm(
        modelOne: sklearn.svm._classes,
        modelTwo: sklearn.svm._classes,
        training_data: list
) -> bool:
    """
    Checks if two models are equal in the sense:
    1. They have the same hyper parameters
    2. They are fitted equally
    3. They make the same predictions
    """
    have_same_params = equal_hyperparams(modelOne, modelTwo) and equal_fitted(modelOne, modelTwo)
    are_predictions_the_same = np.array_equal(modelOne.predict(training_data), modelTwo.predict(training_data))
    return have_same_params and are_predictions_the_same

def equal_hyperparams(
        modelOne: sklearn.svm._classes,
        modelTwo: sklearn.svm._classes
) -> bool:
    """
    Checks if two models have the same hyperparameters
    """
    are_models_the_same_type = type(modelOne) == type(modelTwo)
    do_the_models_have_the_same_parameters = modelOne.get_params() == modelTwo.get_params()

    return are_models_the_same_type and do_the_models_have_the_same_parameters

def equal_fitted(
        modelOne: sklearn.svm._classes,
        modelTwo: sklearn.svm._classes
) -> bool:
    try:
        return (
            np.array_equal(modelOne.support_vectors_, modelTwo.support_vectors_) and
            np.array_equal(modelOne.dual_coef_, modelTwo.dual_coef_) and
            np.array_equal(modelOne.intercept_, modelTwo.intercept_)
        )
    except AttributeError:
        return False