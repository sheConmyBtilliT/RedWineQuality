#cleaning missing values and scaling the features for the models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    """
    Handles missing values in the DataFrame by filling them with the median.
    """
    print("🕵️‍♀️ Checking for missing values...")
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Filling with median...")
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    else:
        print("✅ No missing values found.")
    return df


def prepare_data_for_modeling(df):
    """
    Splits data into features (x) and target (y), and scales the features.
    """
    print("⚙️ Preparing data for modeling...")
    # Separate features (X) and target (y)
    x = df.drop('quality', axis=1)
    y = df['quality']

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("✅ Data prepared and split successfully!")
    return x_train_scaled, x_test_scaled, y_train, y_test