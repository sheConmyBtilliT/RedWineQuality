# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from data_loader import load_data
#from preprocessor import preprocess_data
#from models import get_knn_model, get_svm_model, get_ann_model
#from trainer import train_and_evaluate
from visualizer import plot_quality_distribution

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The main execution block of your script
if __name__ == '__main__':
    # Define the path to your CSV file
    path_to_file = 'winequality-red.csv'

    # Call the function to load the data
    pd.set_option('display.max_columns', None)
    my_dataframe = load_data(path_to_file)

    # Optional: Print the first few rows to confirm it loaded correctly
    if my_dataframe is not None:
        print("Data loaded successfully! Here's a preview:")
        print(my_dataframe.head())