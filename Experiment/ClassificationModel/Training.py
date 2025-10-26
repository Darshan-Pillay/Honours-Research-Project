from ClassificationModel.SplitData import load_training_data_from_pickle_object

from Preprocessing.Utilities import (
    save_model_to_disk
)

from sklearn.svm import (
    SVC
)

def train_model():
    """Trains model and saves it to disk"""
    # Define SVM classifier
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        class_weight='balanced',
        probability=True
    )

    # Get training data pairs
    features, labels = load_training_data_from_pickle_object()

    # Train model
    svm_model.fit(features, labels)

    # Save the model to disk
    save_model_to_disk(
        destination_path="svm_classifier.joblib",
        model=svm_model
    )