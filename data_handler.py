import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import io


def load_and_describe_data(filepath):
    """Loads a CSV and returns the DataFrame and a descriptive string."""
    try:
        df = pd.read_csv(filepath)
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = f"Dataset Loaded Successfully!\n\nShape: {df.shape}\n\nFirst 5 Rows:\n{df.head().to_string()}\n\nData Info:\n{buffer.getvalue()}"
        return df, info_str
    except Exception as e:
        return None, f"An error occurred: {e}"


def preprocess_data(raw_df, log_callback):
    """Performs full data preprocessing for model training."""
    log_callback("Preprocessing data...\n")
    df = raw_df.copy()

    if df.isnull().sum().any():
        log_callback("Handling missing values...\n")
        df = df.fillna(df.median())

    df['quality_category'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    df = df.drop('quality', axis=1)

    X = df.drop('quality_category', axis=1)
    y = df['quality_category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    log_callback(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.\n")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    log_callback("Data scaled using StandardScaler.\n")

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    log_callback(f"Training data balanced with SMOTE. New size: {len(X_train)} samples.\n\n")

    return X_train, X_test, y_train, y_test, scaler


def preprocess_for_tuning(raw_df, log_callback):
    """Performs preprocessing specifically for the tuning process."""
    log_callback("Preprocessing data for tuning...\n")
    df = raw_df.copy()
    if df.isnull().sum().any(): df = df.fillna(df.median())
    df['quality_category'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    df = df.drop('quality', axis=1)
    X = df.drop('quality_category', axis=1)
    y = df['quality_category']
    X_train_tune, _, y_train_tune, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    scaler_tune = StandardScaler()
    X_train_tune = scaler_tune.fit_transform(X_train_tune)
    X_train_tuned, y_train_tuned = SMOTE(random_state=42).fit_resample(X_train_tune, y_train_tune)
    log_callback("Preprocessing for tuning complete.\n\n")
    return X_train_tuned, y_train_tuned


def get_feature_info():
    """Returns the dictionary of feature ranges and defaults for the UI."""
    return {
        'fixed acidity': {'min': 4.0, 'max': 16.0, 'default': 7.4},
        'volatile acidity': {'min': 0.1, 'max': 1.6, 'default': 0.7},
        'citric acid': {'min': 0.0, 'max': 1.0, 'default': 0.0},
        'residual sugar': {'min': 0.9, 'max': 16.0, 'default': 1.9},
        'chlorides': {'min': 0.01, 'max': 0.6, 'default': 0.076},
        'free sulfur dioxide': {'min': 1.0, 'max': 72.0, 'default': 11.0},
        'total sulfur dioxide': {'min': 6.0, 'max': 289.0, 'default': 34.0},
        'density': {'min': 0.990, 'max': 1.004, 'default': 0.9978},
        'pH': {'min': 2.7, 'max': 4.0, 'default': 3.51},
        'sulphates': {'min': 0.3, 'max': 2.0, 'default': 0.56},
        'alcohol': {'min': 8.0, 'max': 15.0, 'default': 9.4}
    }
