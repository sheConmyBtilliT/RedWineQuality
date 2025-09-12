# svm_model.py

from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def train_and_evaluate_svm(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates a Support Vector Machine model.
    """
    print("--- Support Vector Machine (SVM) ---")

    # Initialize and train the model. C and kernel are key hyperparameters.
    # probability=True is needed for predict_proba and is computationally expensive.
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(x_train, y_train)

    # Make predictions
    y_pred = svm.predict(x_test)
    y_pred_proba = svm.predict_proba(x_test)[:, 1] # For AUC score

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['bad', 'good']))
    print("----------------------------------\n")

    return svm