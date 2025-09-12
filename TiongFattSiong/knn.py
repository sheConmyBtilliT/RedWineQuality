# knn_model.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def train_and_evaluate_knn(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates a K-Nearest Neighbors model.
    """
    print("--- K-Nearest Neighbors (KNN) ---")

    # Initialize and train the model
    knn = KNeighborsClassifier(n_neighbors=5) # n_neighbors is a key hyperparameter
    knn.fit(x_train, y_train)

    # Make predictions
    y_pred = knn.predict(x_test)
    y_pred_proba = knn.predict_proba(x_test)[:, 1] # For AUC score

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['bad', 'good']))
    print("---------------------------------\n")

    return knn
