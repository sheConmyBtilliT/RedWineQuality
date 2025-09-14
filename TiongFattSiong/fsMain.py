# main.py

# Import functions from your specialized files
from load_data import load_data
from data_preprocessing import prepare_data, split_and_scale_data, balance_data
from data_visualizer import generate_visualizations

# Import your model functions
from knn import train_and_evaluate_knn
from svm import train_and_evaluate_svm
from ann import train_and_evaluate_ann

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Load Data
    raw_df = load_data()

    if raw_df is not None:
        # 2. Prepare the data (clean, create target variable)
        X, y = prepare_data(raw_df)

        # 2.5 Generate and save data visualizations (Restored step)
        generate_visualizations(raw_df, y)

        # 3. Split data into training and testing sets, and scale features
        X_train, X_test, y_train, y_test = split_and_scale_data(X, y, test_size=0.3)

        # 4. Balance the training data to handle class imbalance
        X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

        # 5. Train and Evaluate each model on the BALANCED training data
        print("\n--- Starting Model Training and Evaluation ---")
        print("Models are trained on balanced data and evaluated on the original test set.\n")

        knn_model = train_and_evaluate_knn(X_train_balanced, y_train_balanced, X_test, y_test)
        svm_model = train_and_evaluate_svm(X_train_balanced, y_train_balanced, X_test, y_test)
        ann_model = train_and_evaluate_ann(X_train_balanced, y_train_balanced, X_test, y_test)

        print("--- Model Evaluation Complete ---")
