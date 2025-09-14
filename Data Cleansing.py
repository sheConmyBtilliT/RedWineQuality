# --- 1. Import Necessary Libraries ---
# Make sure you have these libraries installed. You can install them using pip:
# pip install pandas numpy matplotlib seaborn scikit-learn
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 2. Load the Dataset ---
# The script assumes 'winequality-red.csv' is in the same directory.
try:
    wine_df = pd.read_csv('winequality-red.csv')
    print("Dataset loaded successfully!")
    print("First 5 rows of the dataset:")
    print(wine_df.head())
    print("\n" + "="*50 + "\n")
except FileNotFoundError:
    print("Error: 'winequality-red.csv' not found. Please make sure the file is in the same directory as the script.")
    exit()


# --- 3. Data Cleaning and Exploratory Data Analysis (EDA) ---
print("Step 3: Data Cleaning and EDA")

# Get a summary of the dataset (data types, non-null counts)
print("Dataset Information:")
wine_df.info()
print("\n" + "="*50 + "\n")


# Check for missing values
print("Checking for missing values:")
print(wine_df.isnull().sum())
# This dataset is famously clean, so we don't expect any missing values.
print("No missing values found. The data is clean.\n")
print("\n" + "="*50 + "\n")

# Get descriptive statistics
print("Descriptive Statistics:")
print(wine_df.describe())
print("\n" + "="*50 + "\n")


# --- 4. Data Visualization ---
print("Step 4: Visualizing Data")

# Let's see the distribution of the target variable 'quality'
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=wine_df)
plt.title('Distribution of Red Wine Quality')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.show()
print("The 'quality' scores are mostly concentrated around 5 and 6.")

# Correlation heatmap to see how features relate to each other
plt.figure(figsize=(12, 8))
correlation = wine_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Wine Features')
plt.show()
print("From the heatmap, we can see 'alcohol', 'sulphates', and 'citric acid' have a positive correlation with quality.\n")


# --- 5. Data Preprocessing ---
print("Step 5: Data Preprocessing")

# The 'quality' column has values from 3 to 8. This is a multi-class problem.
# To simplify, we can convert it into a binary classification problem:
# 'good' (1) if quality >= 6
# 'bad'  (0) if quality < 6
# This is a common approach for this dataset.
wine_df['quality_category'] = wine_df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Now, let's see the new distribution
print("Distribution of Quality Category (1: Good, 0: Bad):")
print(wine_df['quality_category'].value_counts())

# Separate the features (X) from the target (y)
X = wine_df.drop(['quality', 'quality_category'], axis=1)
y = wine_df['quality_category']

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("\n" + "="*50 + "\n")


# --- 6. Train-Test Split ---
print("Step 6: Splitting data into training and testing sets")

# We'll use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 'stratify=y' ensures that the train and test sets have a similar proportion of quality categories.

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("\n" + "="*50 + "\n")


# --- 7. Feature Scaling ---
print("Step 7: Scaling the features")
# Scaling is important for many ML algorithms to perform well.
# It standardizes the features to have a mean of 0 and a standard deviation of 1.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features have been scaled successfully.")
print("\n" + "="*50 + "\n")

# --- 8. Model Training & Comparison ---
print("Step 8: Training and Comparing Models")

# We will define a dictionary of models to train and evaluate
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

# Loop through each model to train, predict, and evaluate
for name, model in models.items():
    print(f"--- Training and Evaluating {name} ---")

    # Train the model
    model.fit(X_train_scaled, y_train)
    print(f"{name} training complete.")

    # Make predictions on the scaled test data
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy for {name}: {accuracy:.4f}")

    # Display a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad Quality', 'Good Quality']))

    # Display the confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad Quality', 'Good Quality'], yticklabels=['Bad Quality', 'Good Quality'])
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    print("\n" + "="*50 + "\n")

    # --- 9. Save the Best Model ---
    print("Step 9: Saving the best model (Random Forest)")

    # To prepare the model for deployment, we will train it on the entire dataset.
    # First, create and fit the scaler on the full dataset.
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    # Now, train the final model on all the scaled data.
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_scaled, y)
    print("Final model trained on the entire dataset.")

    # Save the trained model and the scaler to files
    joblib.dump(final_model, 'random_forest_model.joblib')
    joblib.dump(final_scaler, 'scaler.joblib')

    print("\nModel and scaler have been saved successfully as 'random_forest_model.joblib' and 'scaler.joblib'.")
    print("\n--- End of Script ---")