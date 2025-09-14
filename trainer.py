# contains the logic for training and evaluating each model
import numpy as np
from sklearn.metrics import classification_report


def train_and_evaluate(model, model_name, x_train, y_train, x_test, y_test):
    """
    Trains the model and evaluates its performance.
    """
    print(f"\n--- Training {model_name} ---")

    if model_name == 'ANN':
        # For ANN, we need to adjust labels if they don't start from 0
        y_train_ann = y_train - y_train.min()
        y_test_ann = y_test - y_test.min()
        model.fit(x_train, y_train_ann, epochs=50, batch_size=32, verbose=0)
        y_pred_proba = model.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis=1) + y_train.min()
    else:  # For scikit-learn models
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

    print(f"✅ {model_name} trained successfully.")

    # Evaluate the model
    print(f"\n--- Evaluation Report for {model_name} ---")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    print("---------------------------------------\n")
