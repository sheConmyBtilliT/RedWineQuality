import pandas as pd
import numpy as np

from data_loader import load_data
from preprocessor import handle_missing_values, prepare_data_for_modeling
from visualizer import plot_quality_distribution
from knn import get_knn_model
from svm import get_svm_model
from ann import get_ann_model
from trainer import train_and_evaluate


def main():
    """Main function to run the ML pipeline."""
    # Set display options for better output
    pd.set_option('display.max_columns', None)

    # 1. Load Data
    filepath = 'winequality-red.csv'
    df = load_data(filepath)
    if df is None:
        return

    # 2. Clean Data by Handling Missing Values
    df_cleaned = handle_missing_values(df)

    # 3. Visualize Data Distribution
    plot_quality_distribution(df_cleaned)

    # 4. Prepare Data for Modeling (Scaling and Splitting)
    x_train, x_test, y_train, y_test = prepare_data_for_modeling(df_cleaned)

    # 5. Initialize Models
    input_dim = x_train.shape[1]
    output_dim = len(np.unique(y_train))

    models = {
        "k-NN": get_knn_model(),
        "SVM": get_svm_model(),
        "ANN": get_ann_model(input_dim=input_dim, output_dim=output_dim)
    }

    # 6. Train and Evaluate each model
    for name, model in models.items():
        train_and_evaluate(model, name, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
