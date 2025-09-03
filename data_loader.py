#loading the data from the CSV file
import pandas as pd


def load_data(filepath):
    """
    Loads a dataset from a specified CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath, sep=';')
        print("✅ Data loaded successfully!")

        # Display the first 5 rows
        print("--- Dataset Preview ---")
        print(df.head())
        print("-----------------------")

        return df

    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found. Please check the path.")
        return None
