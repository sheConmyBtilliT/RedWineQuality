import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import threading
import os

# Import for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# --- Main Application Class ---
class MachineLearningGUI(ttk.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("Comprehensive Wine Quality Analysis Tool")
        self.geometry("1000x900")

        # --- Class Attributes ---
        self.raw_df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = None
        self.trained_models = {}  # To store trained model objects
        self.best_params = {}  # To store best hyperparameters
        self.model_metrics = {}  # To store performance metrics for charting
        self.model_reports = {}  # To store full classification reports for detailed charts
        self.model_predictions = {}  # To store model predictions for confusion matrix chart

        # --- UI Setup ---
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=BOTH, expand=True)

        # Create a Notebook (tabbed interface)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=BOTH, expand=True, pady=10)

        # --- Tab 1: Data Overview ---
        data_tab = ttk.Frame(notebook, padding="10")
        notebook.add(data_tab, text="1. Data Overview")
        self.setup_data_tab(data_tab)

        # --- Tab 2: Data Visualization ---
        viz_tab = ttk.Frame(notebook, padding="10")
        notebook.add(viz_tab, text="2. Data Visualization")
        self.setup_visualization_tab(viz_tab)

        # --- Tab 3: Model Training & Evaluation ---
        training_tab = ttk.Frame(notebook, padding="10")
        notebook.add(training_tab, text="3. Model Training & Evaluation")
        self.setup_training_tab(training_tab)

        # --- Tab 4: Performance Comparison ---
        comparison_tab = ttk.Frame(notebook, padding="10")
        notebook.add(comparison_tab, text="4. Performance Comparison")
        self.setup_comparison_tab(comparison_tab)

        # --- Tab 5: Live Prediction ---
        prediction_tab = ttk.Frame(notebook, padding="10")
        notebook.add(prediction_tab, text="5. Live Prediction")
        self.setup_prediction_tab(prediction_tab)

    # --- Setup for Tab 1 ---
    def setup_data_tab(self, parent_tab):
        load_button = ttk.Button(parent_tab, text="Load Wine Quality Dataset", command=self.load_data,
                                 bootstyle="primary")
        load_button.pack(pady=10, fill=X)
        self.data_info_text = tk.Text(parent_tab, height=35, width=100, font=("Courier New", 10))
        self.data_info_text.pack(pady=10, fill=BOTH, expand=True)
        self.data_info_text.insert(END, "Please load the 'winequality-red.csv' dataset to begin.")
        self.data_info_text.config(state=DISABLED)

    # --- Setup for Tab 2 ---
    def setup_visualization_tab(self, parent_tab):
        # --- Controls for plotting ---
        plot_controls_frame = ttk.Frame(parent_tab)
        plot_controls_frame.pack(fill=X, pady=5)
        plot_controls_frame.columnconfigure((0, 1, 2), weight=1)

        # Chart generation buttons
        heatmap_button = ttk.Button(plot_controls_frame, text="Correlation Heatmap",
                                    command=self.generate_correlation_heatmap, bootstyle="info")
        heatmap_button.grid(row=0, column=0, sticky=EW, pady=5, padx=(0, 5))
        dist_button = ttk.Button(plot_controls_frame, text="Quality Distributions",
                                 command=self.generate_quality_distributions, bootstyle="info")
        dist_button.grid(row=0, column=1, sticky=EW, pady=5, padx=5)
        boxplots_button = ttk.Button(plot_controls_frame, text="Feature Box Plots",
                                     command=self.generate_feature_boxplots, bootstyle="info")
        boxplots_button.grid(row=0, column=2, sticky=EW, pady=5, padx=(5, 0))

        # Use a ScrolledFrame to make the content scrollable
        self.viz_chart_frame = ScrolledFrame(parent_tab, autohide=True)
        self.viz_chart_frame.pack(fill=BOTH, expand=True, pady=10)

    # --- Setup for Tab 3 ---
    def setup_training_tab(self, parent_tab):
        control_frame = ttk.Frame(parent_tab)
        control_frame.pack(fill=X, pady=5)
        control_frame.columnconfigure((0, 1), weight=1)

        # New button for hyperparameter tuning
        tune_button = ttk.Button(control_frame, text="Tune Hyperparameters (Optional, Slow)",
                                 command=self.start_tuning_thread, bootstyle="warning")
        tune_button.grid(row=0, column=0, sticky=EW, padx=(0, 5), pady=5)

        train_button = ttk.Button(control_frame, text="Start Data Preprocessing & Model Training",
                                  command=self.start_training_thread, bootstyle="success")
        train_button.grid(row=0, column=1, sticky=EW, padx=(5, 0), pady=5)

        self.progress_bar = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=EW, pady=5)

        self.results_text = tk.Text(parent_tab, height=30, width=100, font=("Courier New", 10), wrap="word")
        self.results_text.pack(pady=10, fill=BOTH, expand=True)
        self.results_text.insert(END, "Training results will be displayed here.")
        self.results_text.config(state=DISABLED)

    # --- Setup for Tab 4 (Comparison Chart) ---
    def setup_comparison_tab(self, parent_tab):
        self.comparison_frame = ttk.Frame(parent_tab)
        self.comparison_frame.pack(fill=BOTH, expand=True)

        # --- Controls for plotting ---
        plot_controls_frame = ttk.Frame(self.comparison_frame)
        plot_controls_frame.pack(fill=X, pady=5)
        plot_controls_frame.columnconfigure((0, 1, 2), weight=1)

        # Dropdown for model selection
        self.model_selector = ttk.Combobox(plot_controls_frame, state="readonly", bootstyle="info")
        self.model_selector.grid(row=0, column=0, columnspan=3, sticky=EW, padx=(0, 5), pady=(0, 5))

        # Chart generation buttons
        overall_chart_button = ttk.Button(plot_controls_frame, text="Overall Comparison",
                                          command=self.generate_overall_comparison_chart, bootstyle="info")
        overall_chart_button.grid(row=1, column=0, sticky=EW, pady=5, padx=(0, 5))
        detailed_chart_button = ttk.Button(plot_controls_frame, text="Detailed Report",
                                           command=self.generate_detailed_chart, bootstyle="info")
        detailed_chart_button.grid(row=1, column=1, sticky=EW, pady=5, padx=5)
        avp_chart_button = ttk.Button(plot_controls_frame, text="Actual vs. Predicted",
                                      command=self.generate_actual_vs_predicted_chart, bootstyle="info")
        avp_chart_button.grid(row=1, column=2, sticky=EW, pady=5, padx=(5, 0))

        # Use a ScrolledFrame to make the content scrollable
        self.chart_frame = ScrolledFrame(self.comparison_frame, autohide=True)
        self.chart_frame.pack(fill=BOTH, expand=True, pady=10)

    # --- Setup for Tab 5 (Prediction) ---
    def setup_prediction_tab(self, parent_tab):
        self.prediction_frame = ScrolledFrame(parent_tab, autohide=True)
        self.prediction_frame.pack(fill=BOTH, expand=True, pady=20, padx=20)
        self.prediction_placeholder = ttk.Label(self.prediction_frame,
                                                text="Train a model in Tab 3 to enable prediction.",
                                                font=("Helvetica", 14))
        self.prediction_placeholder.pack()

    def populate_prediction_tab(self):
        # Clear placeholder and prepare for new widgets
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()

        self.feature_info = {
            'fixed acidity': {'min': 4.0, 'max': 16.0, 'default': 7.4},
            'volatile acidity': {'min': 0.1, 'max': 1.6, 'default': 0.7},
            'citric acid': {'min': 0.0, 'max': 1.0, 'default': 0.0},
            'residual sugar': {'min': 0.9, 'max': 16.0, 'default': 1.9},
            'chlorides': {'min': 0.01, 'max': 0.6, 'default': 0.076},
            'free sulfur dioxide': {'min': 1.0, 'max': 72.0, 'default': 11.0},
            'total sulfur dioxide': {'min': 6.0, 'max': 289.0, 'default': 34.0},
            'density': {'min': 0.990, 'max': 1.004, 'default': 0.9978},
            'pH': {'min': 2.7, 'max': 4.0, 'default': 3.51}, 'sulphates': {'min': 0.3, 'max': 2.0, 'default': 0.56},
            'alcohol': {'min': 8.0, 'max': 15.0, 'default': 9.4}
        }

        self.sliders = {}
        self.entries = {}

        # This inner frame will be placed inside the ScrolledFrame
        input_frame = ttk.Frame(self.prediction_frame)
        input_frame.pack(fill=X, expand=True, pady=10)

        for i, (feature, info) in enumerate(self.feature_info.items()):
            # Feature Label
            label = ttk.Label(input_frame, text=f"{feature.replace('_', ' ').title()}", font=("Helvetica", 13))
            label.grid(row=i, column=0, sticky=tk.W, pady=10)

            # Slider
            slider = ttk.Scale(input_frame, from_=info['min'], to=info['max'], orient=HORIZONTAL)
            slider.set(info['default'])
            slider.grid(row=i, column=1, sticky=tk.EW, padx=10)
            self.sliders[feature] = slider

            # Entry box for precise input
            entry_var = tk.StringVar()
            entry = ttk.Entry(input_frame, textvariable=entry_var, width=10, font=("Helvetica", 13))
            entry.grid(row=i, column=2, sticky=tk.W, padx=5)
            self.entries[feature] = entry_var

            # Link slider and entry
            slider.config(command=lambda val, f=feature: self.update_entry_from_slider(f, val))
            entry.bind("<KeyRelease>", lambda event, f=feature: self.update_slider_from_entry(f, event))

            # Initialize entry text
            self.update_entry_from_slider(feature, info['default'])

        input_frame.columnconfigure(1, weight=1)

        # Predict Button
        predict_button = ttk.Button(input_frame, text="Predict Quality", command=self.predict_quality,
                                    bootstyle="success-outline", padding=15)
        predict_button.grid(row=len(self.feature_info), column=0, columnspan=3, pady=30, sticky=tk.EW)

        # Results Display
        results_frame = ttk.LabelFrame(input_frame, text="Prediction Result", padding="20", bootstyle=INFO)
        results_frame.grid(row=len(self.feature_info) + 1, column=0, columnspan=3, pady=10, sticky=tk.EW)
        results_frame.columnconfigure(0, weight=1)

        self.result_label = ttk.Label(results_frame, text="--", font=("Helvetica", 36, "bold"))
        self.result_label.pack(pady=10)

        self.confidence_label = ttk.Label(results_frame, text="", font=("Helvetica", 16))
        self.confidence_label.pack(pady=5)

    def update_entry_from_slider(self, feature, value):
        """Updates the entry box text when the slider is moved."""
        val = float(value)
        if feature in ['chlorides', 'density']:
            self.entries[feature].set(f"{val:.4f}")
        else:
            self.entries[feature].set(f"{val:.2f}")

    def update_slider_from_entry(self, feature, event):
        """Updates the slider position when text is typed in the entry box."""
        try:
            val = float(self.entries[feature].get())
            # Clamp the value to the slider's range to prevent errors
            info = self.feature_info[feature]
            if info['min'] <= val <= info['max']:
                self.sliders[feature].set(val)
        except (ValueError, tk.TclError):
            # Ignore errors from incomplete typing (e.g., "1.")
            pass

    # --- Backend Logic ---
    def load_data(self):
        filepath = filedialog.askopenfilename(title="Select the winequality-red.csv file",
                                              filetypes=(("CSV Files", "*.csv"), ("All files", "*.*")))
        if not filepath: return
        try:
            self.raw_df = pd.read_csv(filepath)
            info_str = f"Dataset Loaded Successfully!\n\nShape: {self.raw_df.shape}\n\nFirst 5 Rows:\n{self.raw_df.head().to_string()}\n\nData Info:\n"
            import io;
            buffer = io.StringIO();
            self.raw_df.info(buf=buffer);
            info_str += buffer.getvalue()
            self.data_info_text.config(state=NORMAL);
            self.data_info_text.delete(1.0, END);
            self.data_info_text.insert(END, info_str);
            self.data_info_text.config(state=DISABLED)
            messagebox.showinfo("Success",
                                "Data loaded. You can now explore visualizations in the 'Data Visualization' tab.")
        except Exception as e:
            messagebox.showerror("Error Loading Data", f"An error occurred: {e}")

    def start_training_thread(self):
        if self.raw_df is None: messagebox.showwarning("No Data",
                                                       "Please load the dataset first in the 'Data Overview' tab."); return
        threading.Thread(target=self.preprocess_and_train, daemon=True).start()

    def start_tuning_thread(self):
        if self.raw_df is None: messagebox.showwarning("No Data", "Please load the dataset first."); return
        if not self.X_train:  # Check if data has been preprocessed
            messagebox.showinfo("Info", "Data will be preprocessed before tuning.")
        threading.Thread(target=self.tune_hyperparameters, daemon=True).start()

    def _preprocess_data_if_needed(self):
        """Internal helper to preprocess data without training models."""
        if self.X_train is not None:
            self.update_results("Data is already preprocessed. Using existing splits.\n")
            return True
        self.update_results("--- Preprocessing Data for Tuning ---\n")
        try:
            df = self.raw_df.copy()
            if df.isnull().sum().any(): self.update_results("Handling missing values...\n"); df = df.fillna(df.median())
            df['quality_category'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0);
            df = df.drop('quality', axis=1)
            X = df.drop('quality_category', axis=1);
            y = df['quality_category']
            # Use a temporary train/test split for tuning, will be overwritten by main training
            X_train_tune, _, y_train_tune, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            scaler_tune = StandardScaler();
            X_train_tune = scaler_tune.fit_transform(X_train_tune)
            smote = SMOTE(random_state=42);
            self.X_train_tuned, self.y_train_tuned = smote.fit_resample(X_train_tune, y_train_tune)
            self.update_results("Preprocessing for tuning complete.\n\n")
            return True
        except Exception as e:
            messagebox.showerror("Preprocessing Error", f"An error occurred during preprocessing: {e}")
            return False

    def tune_hyperparameters(self):
        self.progress_bar.start()
        if not self._preprocess_data_if_needed():
            self.progress_bar.stop()
            return

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier

        self.update_results("--- Starting Hyperparameter Tuning (This may take several minutes) ---\n")

        # Define parameter grids
        param_grids = {
            "KNN": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
            "SVM": {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
            "ANN": {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
        }

        models = {"KNN": KNeighborsClassifier(), "SVM": SVC(probability=True),
                  "ANN": MLPClassifier(max_iter=500, early_stopping=True, random_state=42)}

        self.best_params.clear()

        for name in models:
            self.update_results(f"--- Tuning {name} ---\n")
            grid_search = GridSearchCV(models[name], param_grids[name], cv=3, scoring='f1', n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train_tuned, self.y_train_tuned)
            self.best_params[name] = grid_search.best_params_
            self.update_results(f"Best parameters for {name}: {grid_search.best_params_}\n")
            self.update_results(f"Best F1-score: {grid_search.best_score_:.4f}\n\n")

        self.progress_bar.stop()
        messagebox.showinfo("Tuning Complete",
                            "Hyperparameter tuning finished. The best parameters have been saved and will be used in the next training run.")

    def preprocess_and_train(self):
        self.progress_bar.start()
        self.update_results("--- Starting Data Preprocessing & Model Training ---\n")
        try:
            df = self.raw_df.copy()
            if df.isnull().sum().any(): self.update_results("Handling missing values...\n"); df = df.fillna(df.median())
            df['quality_category'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0);
            df = df.drop('quality', axis=1)
            X = df.drop('quality_category', axis=1);
            y = df['quality_category']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3,
                                                                                    random_state=42, stratify=y)
            self.update_results(
                f"Data split into {len(self.X_train)} training and {len(self.X_test)} testing samples.\n")
            self.scaler = StandardScaler();
            self.X_train = self.scaler.fit_transform(self.X_train);
            self.X_test = self.scaler.transform(self.X_test)
            self.update_results("Data scaled using StandardScaler.\n")
            smote = SMOTE(random_state=42);
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            self.update_results(f"Training data balanced with SMOTE. New size: {len(self.X_train)} samples.\n\n")

            self.update_results("--- Starting Model Training ---\n")
            if self.best_params:
                self.update_results("Using best parameters found during tuning.\n\n")
            else:
                self.update_results("Using default model parameters.\n\n")

            from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier

            # --- FIX STARTS HERE ---
            # Get default SVM parameters that include probability=True
            svm_params = {'probability': True, 'random_state': 42}
            # Update with tuned parameters if they exist
            if "SVM" in self.best_params:
                svm_params.update(self.best_params["SVM"])
            # --- FIX ENDS HERE ---

            # Use tuned params if available, otherwise use defaults
            models_to_train = {
                "KNN": KNeighborsClassifier(**self.best_params.get("KNN", {})),
                "SVM": SVC(**svm_params), # Use the updated svm_params
                "ANN": MLPClassifier(**self.best_params.get("ANN", {'max_iter': 2000, 'early_stopping': True, 'random_state': 42}))
            }

            self.model_metrics.clear();
            self.model_reports.clear();
            self.model_predictions.clear()
            for name, model in models_to_train.items():
                self.update_results(f"--- Training {name} ---\n")
                self.update_results(f"Parameters: {model.get_params()}\n")
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                y_pred = model.predict(self.X_test)
                self.model_predictions[name] = y_pred  # Store predictions
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                report_dict = classification_report(self.y_test, y_pred, target_names=['bad', 'good'], output_dict=True)
                self.model_reports[name] = report_dict
                accuracy = accuracy_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba)
                f1_good = report_dict['good']['f1-score']
                self.model_metrics[name] = {'Accuracy': accuracy, 'AUC': auc, 'F1-Score (Good)': f1_good}
                report_str = f"Accuracy: {accuracy:.4f}\nAUC Score: {auc:.4f}\nClassification Report:\n{classification_report(self.y_test, y_pred, target_names=['bad', 'good'])}\n"
                self.update_results(report_str)

            self.model_selector['values'] = list(self.trained_models.keys())
            if self.trained_models: self.model_selector.set(list(self.trained_models.keys())[0])

            self.populate_prediction_tab()
            messagebox.showinfo("Success",
                                "All models trained successfully! Proceed to the 'Performance Comparison' or 'Live Prediction' tab.")
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred: {e}")
        finally:
            self.progress_bar.stop()

    def clear_chart_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def autosave_figure(self, fig, filename):
        """Saves the figure to a 'charts' directory and shows a message."""
        charts_dir = 'charts'
        os.makedirs(charts_dir, exist_ok=True)
        filepath = os.path.join(charts_dir, filename)
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Chart Saved", f"Chart automatically saved to:\n{os.path.abspath(filepath)}")
        except Exception as e:
            messagebox.showerror("Error Saving Chart", f"An error occurred: {e}")

    # --- VISUALIZATION PLOTS ---
    def generate_correlation_heatmap(self):
        if self.raw_df is None: messagebox.showwarning("No Data", "Please load the dataset first."); return
        self.clear_chart_frame(self.viz_chart_frame)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        sns.heatmap(self.raw_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap of Wine Features', fontsize=16)
        plt.tight_layout()
        self.autosave_figure(fig, 'correlation_heatmap.png')
        canvas = FigureCanvasTkAgg(fig, master=self.viz_chart_frame);
        canvas.draw();
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def generate_quality_distributions(self):
        if self.raw_df is None: messagebox.showwarning("No Data", "Please load the dataset first."); return
        self.clear_chart_frame(self.viz_chart_frame)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
        df = self.raw_df.copy()
        df['quality_category'] = df['quality'].apply(lambda x: 'Good' if x >= 7 else 'Bad')

        # Plot 1: Original Quality Scores
        sns.countplot(x='quality', data=df, ax=axes[0], hue='quality', palette='viridis', legend=False)
        axes[0].set_title('Distribution of Original Quality Scores')
        for p in axes[0].patches:
            axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                             textcoords='offset points')

        # Plot 2: Good vs. Bad Categories
        sns.countplot(x='quality_category', data=df, ax=axes[1], hue='quality_category', palette='OrRd', legend=False)
        axes[1].set_title('Distribution of Good vs. Bad Categories')
        for p in axes[1].patches:
            axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                             textcoords='offset points')

        plt.tight_layout()
        self.autosave_figure(fig, 'quality_distributions.png')
        canvas = FigureCanvasTkAgg(fig, master=self.viz_chart_frame);
        canvas.draw();
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def generate_feature_boxplots(self):
        if self.raw_df is None: messagebox.showwarning("No Data", "Please load the dataset first."); return
        self.clear_chart_frame(self.viz_chart_frame)
        df = self.raw_df.copy()
        df['quality_category'] = df['quality'].apply(lambda x: 'Good' if x >= 7 else 'Bad')
        features = df.columns.drop(['quality', 'quality_category'])
        num_features = len(features)

        # Rearranged to 2 columns and a taller figure for better scrolling
        fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(15, 24), dpi=100)
        axes = axes.flatten()
        for i, feature in enumerate(features):
            sns.boxplot(x='quality_category', y=feature, data=df, ax=axes[i], palette='Set2')
            axes[i].set_title(f'{feature.title()} vs. Quality', fontsize=12)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
        for i in range(num_features, len(axes)):  # Hide unused subplots
            axes[i].set_visible(False)

        plt.tight_layout(pad=3.0)
        self.autosave_figure(fig, 'feature_boxplots.png')

        # Embed the tall figure into the scrollable frame
        canvas = FigureCanvasTkAgg(fig, master=self.viz_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    # --- COMPARISON PLOTS ---
    def generate_overall_comparison_chart(self):
        if not self.model_metrics: messagebox.showwarning("No Data", "Please train the models first in Tab 3."); return
        self.clear_chart_frame(self.chart_frame)
        metrics_df = pd.DataFrame(self.model_metrics).T.reset_index().rename(columns={'index': 'Model'})
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
        self.autosave_figure(fig, 'overall_performance_comparison.png')
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame);
        canvas.draw();
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def generate_detailed_chart(self):
        selected_model = self.model_selector.get()
        if not selected_model: messagebox.showwarning("No Model Selected",
                                                      "Please select a model from the dropdown."); return
        if not self.model_reports: messagebox.showwarning("No Data", "Please train the models first in Tab 3."); return
        self.clear_chart_frame(self.chart_frame)
        report = self.model_reports[selected_model];
        data = []
        for class_name, metrics in report.items():
            if class_name in ['bad', 'good']:
                data.append({'Class': class_name.title(), 'Metric': 'Precision', 'Score': metrics['precision']})
                data.append({'Class': class_name.title(), 'Metric': 'Recall', 'Score': metrics['recall']})
                data.append({'Class': class_name.title(), 'Metric': 'F1-Score', 'Score': metrics['f1-score']})
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
        self.autosave_figure(fig, f'detailed_report_{selected_model}.png')
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame);
        canvas.draw();
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def generate_actual_vs_predicted_chart(self):
        selected_model = self.model_selector.get()
        if not selected_model: messagebox.showwarning("No Model Selected",
                                                      "Please select a model from the dropdown."); return
        if not self.model_predictions: messagebox.showwarning("No Data",
                                                              "Please train the models first in Tab 3."); return
        self.clear_chart_frame(self.chart_frame)
        actual_labels = self.y_test;
        predicted_labels = self.model_predictions[selected_model]
        class_mapping = {0: 'Bad', 1: 'Good'}
        actual_text = pd.Series(actual_labels).map(class_mapping);
        predicted_text = pd.Series(predicted_labels).map(class_mapping)
        df_actual = pd.DataFrame({'Quality': actual_text, 'Type': 'Actual'});
        df_predicted = pd.DataFrame({'Quality': predicted_text, 'Type': 'Predicted'})
        combined_df = pd.concat([df_actual, df_predicted])
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
        sns.countplot(data=combined_df, x='Type', hue='Quality', ax=ax, palette={'Bad': '#D32F2F', 'Good': '#4CAF50'})
        ax.set_title(f'Actual vs. Predicted Quality Counts for {selected_model}', fontsize=16);
        ax.set_xlabel('Category', fontsize=12);
        ax.set_ylabel('Count', fontsize=12)
        for p in ax.patches: ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                         ha='center', va='center', fontsize=11, color='white', xytext=(0, -12),
                                         textcoords='offset points', weight='bold')
        plt.tight_layout()
        self.autosave_figure(fig, f'actual_vs_predicted_{selected_model}.png')
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame);
        canvas.draw();
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    def update_results(self, text):
        self.results_text.config(state=NORMAL);
        self.results_text.insert(END, text);
        self.results_text.see(END);
        self.results_text.config(state=DISABLED)

    def update_label(self, feature, value):
        if feature in ['chlorides', 'density']:
            formatted_value = f"{float(value):.4f}"
        else:
            formatted_value = f"{float(value):.2f}"
        self.value_labels[feature].config(text=formatted_value)

    def predict_quality(self):
        """Validates input data from entries and runs the prediction."""
        # Use the best model for prediction, defaulting to SVM
        best_model_name = "SVM"
        if self.model_metrics:
            # Find the model with the best F1-Score for the 'Good' class
            best_model_name = max(self.model_metrics, key=lambda k: self.model_metrics[k]['F1-Score (Good)'])

        if best_model_name not in self.trained_models:
            messagebox.showwarning("Model Not Ready",
                                   f"The best model ({best_model_name}) is not trained. Please train it in Tab 3.")
            return

        input_data = []
        error_messages = []
        for feature, entry_var in self.entries.items():
            value_str = entry_var.get()
            info = self.feature_info[feature]
            try:
                value_float = float(value_str)
                if not (info['min'] <= value_float <= info['max']):
                    error_messages.append(
                        f"- {feature.title()}: Value must be between {info['min']} and {info['max']}.")
                else:
                    input_data.append(value_float)
            except ValueError:
                error_messages.append(f"- {feature.title()}: Must be a valid number.")

        if error_messages:
            messagebox.showerror("Invalid Input",
                                 "Please correct the following errors:\n\n" + "\n".join(error_messages))
            return

        try:
            input_df = pd.DataFrame([input_data], columns=self.entries.keys())
            scaled_data = self.scaler.transform(input_df)
            model = self.trained_models[best_model_name]
            prediction_numeric = model.predict(scaled_data)[0]
            prediction_proba = model.predict_proba(scaled_data)[0]
            class_mapping = {0: 'BAD', 1: 'GOOD'}
            result_text = class_mapping.get(prediction_numeric, 'UNKNOWN')
            confidence = prediction_proba[prediction_numeric] * 100
            self.result_label.config(text=result_text, bootstyle=SUCCESS if result_text == 'GOOD' else DANGER)
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}% (using {best_model_name})")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred during prediction: {e}")


# --- Run the Application ---
if __name__ == "__main__":
    app = MachineLearningGUI()
    app.mainloop()

