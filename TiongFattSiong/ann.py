# ann_model.py

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def train_and_evaluate_ann(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates an Artificial Neural Network model.
    """
    print("--- Artificial Neural Network (ANN) ---")

    # Initialize and train the model. Many hyperparameters can be tuned here.
    ann = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                        max_iter=1000, random_state=42, early_stopping=True)
    ann.fit(x_train, y_train)

    # Make predictions
    y_pred = ann.predict(x_test)
    y_pred_proba = ann.predict_proba(x_test)[:, 1] # For AUC score

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['bad', 'good']))
    print("-----------------------------------\n")

    return ann
