import matplotlib.pyplot as plt
import seaborn as sns # Seaborn is often used with Matplotlib for better aesthetics

def plot_quality_distribution(df):
    """
    Creates and displays a count plot of the wine quality.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='quality', data=df)
    plt.title('Distribution of Red Wine Quality')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    print("📊 Generating plot...")
    plt.show()