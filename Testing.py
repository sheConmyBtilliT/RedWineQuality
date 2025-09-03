# STEP 1: IMPORT LIBRARIES
print("--- STEP 1: Importing Libraries ---")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Libraries imported successfully.\n")


def run_wine_quality_prediction():
    """
    Main function to run the entire wine quality prediction workflow.
    """
    # STEP 2: LOAD AND PREVIEW DATA
    print("--- STEP 2: Loading and Previewing Data ---")
    try:
        df = pd.read_csv('winequality-red.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'winequality-red.csv' not found. Please ensure the file is in the same directory.")
        return

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nStatistical Summary of Features:")
    print(df.describe())

    # --- Exploratory Data Analysis (EDA) ---
    print("\nPerforming Exploratory Data Analysis...")
    # Visualize the distribution of the original 'quality' score
    plt.figure(figsize=(8, 6))
    # Updated line to resolve the FutureWarning
    sns.countplot(x='quality', data=df, palette='viridis', hue='quality', legend=False)
    plt.title('Distribution of Red Wine Quality Scores')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Visualize the correlation between all features
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix of Red Wine Features')
    plt.show()
    print("EDA visualizations are complete.\n")


    # STEP 3: DATA PRE-PROCESSING AND REPRESENTATION
    print("--- STEP 3: Data Pre-processing and Representation ---")
    # Check for missing values
    print("Checking for Missing Values:")
    print(df.isnull().sum())
    print("No missing values found.")

    # Feature Engineering: Binarize the target variable 'quality'
    # We will classify wines as 'good' (quality score > 6) or 'bad' (quality score <= 6).
    # This simplifies the problem into a binary classification task.
    print("\nBinarizing the target variable 'quality' (1 for Good, 0 for Bad)...")
    df['quality'] = df['quality'].apply(lambda x: 1 if x > 6 else 0)
    print("New distribution of 'quality':")
    print(df['quality'].value_counts())

    # Separate features (X) and the target variable (y)
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split data into training and testing sets (80% train, 20% test)
    # We use 'stratify=y' to maintain the same proportion of good/bad wine in both sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split into training and testing sets:")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Normalization: Standardize features to have a mean of 0 and a variance of 1
    # This is crucial for distance-based algorithms like KNN and SVM, and for ANNs.
    print("\nNormalizing data using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data pre-processing complete.\n")


    # STEP 4: TRAIN AND EVALUATE CLASSIFICATION MODELS
    print("--- STEP 4: Training and Evaluating Classification Models ---")
    models = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine (SVM)": SVC(kernel='rbf', random_state=42),
        "Artificial Neural Network (ANN)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)
    }

    # Store results for final comparison
    results = {}

    for name, model in models.items():
        print(f"\n----- Training {name} -----")
        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model's performance
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Bad Quality', 'Good Quality'], output_dict=True)

        results[name] = {
            'Accuracy': accuracy,
            'Precision': report['Good Quality']['precision'],
            'Recall': report['Good Quality']['recall'],
            'F1-Score': report['Good Quality']['f1-score']
        }

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Bad Quality', 'Good Quality']))

        # Visualize the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bad Quality', 'Good Quality'],
                    yticklabels=['Bad Quality', 'Good Quality'])
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    print("Model training and evaluation complete.\n")


    # STEP 5: COMPARE MODEL PERFORMANCES
    print("--- STEP 5: Comparing Model Performances ---")
    results_df = pd.DataFrame(results).T.reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)
    results_df = results_df.sort_values(by='Accuracy', ascending=False)

    print("Summary of Model Performance Metrics:")
    print(results_df)

    # Visualize the comparison of the main evaluation metrics
    results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=results_melted, palette='magma')
    plt.title('Comparison of Classification Models')
    plt.ylabel('Score')
    plt.ylim(0.7, 1.0) # Adjust y-axis for better visualization
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=15)
    # Add score values on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge')
    plt.tight_layout()
    plt.show()

    best_model_name = results_df.iloc[0]['Model']
    best_model_accuracy = results_df.iloc[0]['Accuracy']
    print(f"\nThe best performing model based on accuracy is the '{best_model_name}' with an accuracy of {best_model_accuracy:.4f}.")


if __name__ == '__main__':
    run_wine_quality_prediction()

