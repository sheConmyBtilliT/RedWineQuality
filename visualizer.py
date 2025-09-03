# function for plotting the data distribution
import matplotlib.pyplot as plt
import seaborn as sns


def plot_quality_distribution(df):
    """
    Creates and displays a count plot of the wine quality.
    """

    print("DEBUG: Columns in DataFrame are:", df.columns.tolist())

    plt.figure(figsize=(10, 6))
    sns.countplot(x='quality', data=df, palette='viridis')
    plt.title('Distribution of Red Wine Quality', fontsize=16)
    plt.xlabel('Quality Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    print("📊 Generating plot of wine quality distribution...")
    plt.show()
