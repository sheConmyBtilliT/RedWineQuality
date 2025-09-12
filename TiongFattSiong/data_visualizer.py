import matplotlib.pyplot as plt
import seaborn as sns


def generate_visualizations(original_df, target_y):
    """
    Generates and saves key visualizations for the dataset.

    Args:
        original_df (pd.DataFrame): The original dataframe with the 'quality' column.
        target_y (pd.Series): The prepared binary target variable.
    """
    print("\n--- Generating Data Visualizations ---")

    # 1. Original Quality Score Distribution (NEW PLOT)
    try:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='quality', data=original_df, hue='quality', palette='viridis', legend=False)
        plt.title('Distribution of Original Wine Quality Scores', fontsize=16)
        plt.xlabel('Quality Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig('quality_score_distribution.png')
        plt.close()
        print("✅ Original quality score distribution saved as 'quality_score_distribution.png'")
    except Exception as e:
        print(f"Could not generate quality score distribution plot. Error: {e}")

    # 2. Correlation Heatmap
    try:
        plt.figure(figsize=(12, 10))
        correlation_matrix = original_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap of Red Wine Features', fontsize=16)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")
    except Exception as e:
        print(f"Could not generate correlation heatmap. Error: {e}")

    # 3. Class Distribution Pie Chart (after categorization)
    try:
        plt.figure(figsize=(8, 8))
        target_y.value_counts().plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            colors=['#d9534f', '#5cb85c'],
            labels=['Bad Quality', 'Good Quality'],
            wedgeprops={'edgecolor': 'white'}
        )
        plt.title('Distribution of Wine Quality Target Variable', fontsize=16)
        plt.ylabel('')  # Hides the 'quality_category' label
        plt.tight_layout()
        plt.savefig('quality_distribution_pie.png')
        plt.close()
        print("✅ Quality distribution pie chart saved as 'quality_distribution_pie.png'")
    except Exception as e:
        print(f"Could not generate pie chart. Error: {e}")

    print("------------------------------------")