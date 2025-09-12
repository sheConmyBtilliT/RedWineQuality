import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import necessary functions from your other files
from load_data import load_data
from data_preprocessing import prepare_data, balance_data
from svm import train_and_evaluate_svm


def main():
    """
    Main function to run the focused SVM model training and saving pipeline.
    """
    print("--- Starting Final SVM Model Training Pipeline ---")

    # 1. Load Data
    raw_df = load_data()
    if raw_df is None:
        print("Stopping pipeline: Data could not be loaded.")
        return

    # 2. Prepare Data (Clean, create target variable)
    X, y = prepare_data(raw_df)

    # 3. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. Scale the features and SAVE the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler.joblib')
    print("\n✅ Data scaler has been trained and saved to 'scaler.joblib'")

    # Convert scaled arrays back to DataFrames for the next steps
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 5. Balance the training data using SMOTE
    X_train_balanced, y_train_balanced = balance_data(X_train_df, y_train)

    # 6. Train, Evaluate, and Save the SVM Model
    print("\n--- Training and Evaluating Final SVM Model ---")
    svm_model = train_and_evaluate_svm(X_train_balanced, y_train_balanced, X_test_df, y_test)

    joblib.dump(svm_model, 'svm_model.joblib')
    print("✅ Best performance model (SVM) has been trained and saved to 'svm_model.joblib'")
    print("\n--- Model Training Pipeline Complete ---")
    print("You can now run 'app_final.py' to launch the prediction UI.")


if __name__ == "__main__":
    main()