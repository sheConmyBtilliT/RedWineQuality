import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def prepare_data(df):
    """
    Performs initial data cleaning and feature engineering.
    """
    print("\n--- Starting Data Preparation ---")

    # 1. Handle Missing Values
    if df.isnull().sum().any():
        print("Missing values found! Filling them with the column median...")
        df = df.fillna(df.median())
        print("✅ Missing values handled.")
    else:
        print("✅ No missing values found.")

    # 2. Create target variable: 'quality' >= 7 is 'good' (1), else 'bad' (0)
    # This transforms it into a binary classification problem.
    df['quality_category'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    df = df.drop('quality', axis=1)

    print("✅ Target variable 'quality_category' created ('good': 1, 'bad': 0).")
    print("Original class distribution:")
    print(df['quality_category'].value_counts(normalize=True))

    X = df.drop('quality_category', axis=1)
    y = df['quality_category']
    print("---------------------------------")
    return X, y


def split_and_scale_data(X, y, test_size=0.3, random_state=42):
    """
    Splits the data into training and testing sets, then scales the features.
    """
    # Using stratify=y is important for imbalanced datasets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Convert scaled arrays back to DataFrames to preserve column names
    x_train = pd.DataFrame(x_train_scaled, columns=X.columns)
    x_test = pd.DataFrame(x_test_scaled, columns=X.columns)

    print("\n--- Data Splitting and Scaling ---")
    print("✅ Data split and scaled successfully.")
    print(f"Training set size: {len(x_train)} samples")
    print(f"Testing set size: {len(x_test)} samples")
    print("----------------------------------")

    return x_train, x_test, y_train, y_test


def balance_data(x_train, y_train, random_state=42):
    """
    Balances the training data using SMOTE to address class imbalance.
    This should only be applied to the training data.
    """
    print("\n--- Balancing Training Data ---")
    smote = SMOTE(random_state=random_state)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

    print("✅ Training data balanced with SMOTE.")
    print("Balanced training set target counts:")
    print(y_train_balanced.value_counts())
    print("-------------------------------")

    return x_train_balanced, y_train_balanced