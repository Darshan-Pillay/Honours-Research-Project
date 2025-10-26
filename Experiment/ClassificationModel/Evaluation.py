import matplotlib.pyplot as plt
from scipy.constants import precision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold
)

from sklearn.svm import SVC

from ClassificationModel.SplitData import (
    load_training_data_from_pickle_object, load_test_data_from_pickle_object
)
from Preprocessing.Utilities import load_model_from_disk


def evaluate_model_using_cross_validation():
    training_features, labels = load_training_data_from_pickle_object()

    # SVM Model
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        class_weight='balanced',
        probability=True
    )

    # Use stratified K Fold because our dataset is imbalanced
    cross_validation_strategy = StratifiedKFold(
        n_splits=10,
        random_state=42,
        shuffle=True
    )

    # Metrics
    # scoring = {
    #    'accuracy',
    #    'precision',
    #    'recall',
    #    'f1'
    # }

    results = cross_val_score(
        svm_model,
        training_features,
        labels,
        scoring='accuracy',
        cv=cross_validation_strategy,
        verbose=1
    )

def compute_confusion_matrix():
    """
    Computes confusion matrix for training data
    on our SVM models
    """
    svm_model: SVC = load_model_from_disk(
        model_path="svm_classifier.joblib"
    )

    training_features, labels = load_training_data_from_pickle_object()
    model_predictions = svm_model.predict(training_features)

    confusion = confusion_matrix(labels, model_predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["Healthy", "Diseased"]
    )
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def compute_roc():
    """
    Computes ROC for our training data
    on our SVM models
    """
    svm_model: SVC = load_model_from_disk(
        model_path="svm_classifier.joblib"
    )

    training_features, labels = load_training_data_from_pickle_object()
    display = RocCurveDisplay.from_estimator(svm_model, training_features, labels)

    # Shade region under the ROC curve
    axis = display.ax_

    axis.fill_between(
        display.fpr,
        display.tpr,
        color='skyblue',
        alpha=0.4
    )

    plt.show()

def evaluate_model_for_test_data():
    """
    Prints to console metrics measuring the
    generalization capacity of our model
    """
    test_features, labels = load_test_data_from_pickle_object()

    svm_model = load_model_from_disk(
        model_path="svm_classifier.joblib"
    )

    predictions = svm_model.predict(test_features)

    accuracy = accuracy_score(predictions, labels)
    precision = precision_score(predictions, labels, average="binary")
    recall = recall_score(predictions, labels, average="binary")
    f1 = f1_score(predictions, labels, average="binary")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def confusion_matrix_for_test_data():
    """
    Shows confusion matrix on our test data
    for our model
    """
    svm_model: SVC = load_model_from_disk(
        model_path="svm_classifier.joblib"
    )

    test_features, labels = load_test_data_from_pickle_object()
    model_predictions = svm_model.predict(test_features)

    confusion = confusion_matrix(labels, model_predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=["Healthy", "Diseased"]
    )
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def roc_curve_fors_test_data():
    """
        Computes ROC for our test data
        on our SVM model
        """
    svm_model: SVC = load_model_from_disk(
        model_path="svm_classifier.joblib"
    )

    test_features, labels = load_test_data_from_pickle_object()
    display = RocCurveDisplay.from_estimator(svm_model, test_features, labels)

    # Shade region under the ROC curve
    axis = display.ax_

    axis.fill_between(
        display.fpr,
        display.tpr,
        color='skyblue',
        alpha=0.4
    )

    plt.show()

roc_curve_fors_test_data()
