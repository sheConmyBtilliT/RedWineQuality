import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
import pandas as pd
from tkinter import messagebox
import os


def _clear_chart_frame(frame):
    """Helper function to clear old charts from a frame."""
    for widget in frame.winfo_children():
        widget.destroy()


def _autosave_figure(fig, filename):
    """Saves the figure to a 'charts' directory."""
    charts_dir = 'charts'
    os.makedirs(charts_dir, exist_ok=True)
    filepath = os.path.join(charts_dir, filename)
    try:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        messagebox.showinfo("Chart Saved", f"Chart automatically saved to:\n{os.path.abspath(filepath)}")
    except Exception as e:
        messagebox.showerror("Error Saving Chart", f"An error occurred: {e}")


def _embed_chart_in_frame(fig, parent_frame):
    """Embeds a matplotlib figure into a tkinter frame."""
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# --- Data Visualization Charts ---
def generate_correlation_heatmap(raw_df, parent_frame):
    if raw_df is None: return messagebox.showwarning("No Data", "Please load the dataset first.")
    _clear_chart_frame(parent_frame)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    sns.heatmap(raw_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Wine Features', fontsize=16)
    plt.tight_layout()
    _autosave_figure(fig, 'correlation_heatmap.png')
    _embed_chart_in_frame(fig, parent_frame)


def generate_quality_distributions(raw_df, parent_frame):
    if raw_df is None: return messagebox.showwarning("No Data", "Please load the dataset first.")
    _clear_chart_frame(parent_frame)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    df = raw_df.copy()
    df['quality_category'] = df['quality'].apply(lambda x: 'Good' if x >= 7 else 'Bad')

    plots_data = [
        {'ax': axes[0], 'x': 'quality', 'title': 'Distribution of Original Quality Scores', 'palette': 'viridis'},
        {'ax': axes[1], 'x': 'quality_category', 'title': 'Distribution of Good vs. Bad Categories', 'palette': 'OrRd'}
    ]
    for p_data in plots_data:
        sns.countplot(x=p_data['x'], data=df, ax=p_data['ax'], hue=p_data['x'], palette=p_data['palette'], legend=False)
        p_data['ax'].set_title(p_data['title'])
        for p in p_data['ax'].patches:
            p_data['ax'].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                  ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                  textcoords='offset points')

    plt.tight_layout()
    _autosave_figure(fig, 'quality_distributions.png')
    _embed_chart_in_frame(fig, parent_frame)


def generate_feature_boxplots(raw_df, parent_frame):
    if raw_df is None: return messagebox.showwarning("No Data", "Please load the dataset first.")
    _clear_chart_frame(parent_frame)
    df = raw_df.copy()
    df['quality_category'] = df['quality'].apply(lambda x: 'Good' if x >= 7 else 'Bad')
    features = df.columns.drop(['quality', 'quality_category'])

    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 24), dpi=100)
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.boxplot(x='quality_category', y=feature, data=df, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{feature.title()} vs. Quality', fontsize=12)
        axes[i].set_xlabel('');
        axes[i].set_ylabel('')
    for i in range(len(features), len(axes)): axes[i].set_visible(False)

    plt.tight_layout(pad=3.0)
    _autosave_figure(fig, 'feature_boxplots.png')
    _embed_chart_in_frame(fig, parent_frame)


# --- Performance Comparison Charts ---
def generate_overall_comparison_chart(model_metrics, parent_frame):
    if not model_metrics: return messagebox.showwarning("No Data", "Please train the models first.")
    _clear_chart_frame(parent_frame)
    metrics_df = pd.DataFrame(model_metrics).T.reset_index().rename(columns={'index': 'Model'})
    df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', ax=ax, palette='viridis')
    ax.set_title('Overall Model Performance Comparison', fontsize=16);
    ax.set_xlabel('Model', fontsize=12);
    ax.set_ylabel('Score', fontsize=12);
    ax.set_ylim(0, 1)
    for p in ax.patches: ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                     textcoords='offset points')
    plt.tight_layout()
    _autosave_figure(fig, 'overall_performance_comparison.png')
    _embed_chart_in_frame(fig, parent_frame)


def generate_detailed_chart(selected_model, model_reports, parent_frame):
    if not selected_model or not model_reports: return messagebox.showwarning("No Data",
                                                                              "Please train and select a model.")
    _clear_chart_frame(parent_frame)
    report = model_reports[selected_model];
    data = []
    for class_name, metrics in report.items():
        if class_name in ['bad', 'good']:
            data.extend([{'Class': class_name.title(), 'Metric': 'Precision', 'Score': metrics['precision']},
                         {'Class': class_name.title(), 'Metric': 'Recall', 'Score': metrics['recall']},
                         {'Class': class_name.title(), 'Metric': 'F1-Score', 'Score': metrics['f1-score']}])
    report_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    sns.barplot(data=report_df, x='Class', y='Score', hue='Metric', ax=ax, palette='plasma')
    ax.set_title(f'Detailed Performance Metrics for {selected_model}', fontsize=16);
    ax.set_xlabel('Wine Quality Class', fontsize=12);
    ax.set_ylabel('Score', fontsize=12);
    ax.set_ylim(0, 1)
    for p in ax.patches: ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                     textcoords='offset points')
    plt.tight_layout()
    _autosave_figure(fig, f'detailed_report_{selected_model}.png')
    _embed_chart_in_frame(fig, parent_frame)


def generate_actual_vs_predicted_chart(selected_model, y_test, model_predictions, parent_frame):
    if not selected_model or y_test is None or not model_predictions: return messagebox.showwarning("No Data",
                                                                                                    "Please train and select a model.")
    _clear_chart_frame(parent_frame)
    class_mapping = {0: 'Bad', 1: 'Good'}
    actual_text = pd.Series(y_test).map(class_mapping)
    predicted_text = pd.Series(model_predictions[selected_model]).map(class_mapping)
    combined_df = pd.concat([pd.DataFrame({'Quality': actual_text, 'Type': 'Actual'}),
                             pd.DataFrame({'Quality': predicted_text, 'Type': 'Predicted'})])
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    sns.countplot(data=combined_df, x='Type', hue='Quality', ax=ax, palette={'Bad': '#D32F2F', 'Good': '#4CAF50'})
    ax.set_title(f'Actual vs. Predicted Quality Counts for {selected_model}', fontsize=16);
    ax.set_xlabel('Category', fontsize=12);
    ax.set_ylabel('Count', fontsize=12)
    for p in ax.patches: ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', fontsize=11, color='white', xytext=(0, -12),
                                     textcoords='offset points', weight='bold')
    plt.tight_layout()
    _autosave_figure(fig, f'actual_vs_predicted_{selected_model}.png')
    _embed_chart_in_frame(fig, parent_frame)


def generate_confusion_matrix_chart(selected_model, y_test, model_predictions, parent_frame):
    if not selected_model or y_test is None or not model_predictions: return messagebox.showwarning("No Data",
                                                                                                    "Please train and select a model.")
    _clear_chart_frame(parent_frame)
    cm = confusion_matrix(y_test, model_predictions[selected_model])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
    ax.set_xlabel('Predicted Label', fontsize=12);
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_title(f'Confusion Matrix for {selected_model}', fontsize=16)
    plt.tight_layout()
    _autosave_figure(fig, f'confusion_matrix_{selected_model}.png')
    _embed_chart_in_frame(fig, parent_frame)
