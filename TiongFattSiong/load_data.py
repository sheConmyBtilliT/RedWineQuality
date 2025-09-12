import pandas as pd

def load_data():
    """
    Loads a dataset from a specified CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data, or None if the file is not found.
    """
    filepath = 'winequality-red.csv'

    try:
        df = pd.read_csv(filepath)
        print("✅ Data loaded successfully!")
        # This return statement is the fix.
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None
