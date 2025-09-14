import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
import threading

# Import from our refactored modules
import data_handler
import model_trainer
import visualizer


# --- Main Application Class ---
class MachineLearningGUI(ttk.Window):
    def __init__(self):
        super().__init__(themename="superhero")
        self.title("Comprehensive Wine Quality Analysis Tool")
        self.geometry("1100x900")

        # --- Class Attributes ---
        self.raw_df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.scaler = None
        self.trained_models = {}
        self.best_params = {}
        self.model_metrics = {}
        self.model_reports = {}
        self.model_predictions = {}

        # --- UI Setup ---
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=BOTH, expand=True)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=BOTH, expand=True, pady=10)

        # --- Tab Setup ---
        tabs = {
            "1. Data Overview": self.setup_data_tab,
            "2. Data Visualization": self.setup_visualization_tab,
            "3. Model Training & Evaluation": self.setup_training_tab,
            "4. Performance Comparison": self.setup_comparison_tab,
            "5. Live Prediction": self.setup_prediction_tab
        }
        for text, setup_func in tabs.items():
            tab = ttk.Frame(notebook, padding="10")
            notebook.add(tab, text=text)
            setup_func(tab)

    # --- UI Setup for Each Tab ---
    def setup_data_tab(self, parent_tab):
        load_button = ttk.Button(parent_tab, text="Load Wine Quality Dataset", command=self.load_data,
                                 bootstyle="primary")
        load_button.pack(pady=10, fill=X)
        self.data_info_text = tk.Text(parent_tab, height=35, width=100, font=("Courier New", 10))
        self.data_info_text.pack(pady=10, fill=BOTH, expand=True)
        self.data_info_text.insert(END, "Please load the 'winequality-red.csv' dataset to begin.")
        self.data_info_text.config(state=DISABLED)

    def setup_visualization_tab(self, parent_tab):
        plot_controls_frame = ttk.Frame(parent_tab)
        plot_controls_frame.pack(fill=X, pady=5)
        plot_controls_frame.columnconfigure((0, 1, 2), weight=1)

        buttons = {
            "Correlation Heatmap": self.generate_correlation_heatmap,
            "Quality Distributions": self.generate_quality_distributions,
            "Feature Box Plots": self.generate_feature_boxplots
        }
        for i, (text, command) in enumerate(buttons.items()):
            ttk.Button(plot_controls_frame, text=text, command=command, bootstyle="info").grid(row=0, column=i,
                                                                                               sticky=tk.EW, pady=5,
                                                                                               padx=5 if i > 0 else (0,
                                                                                                                     5))

        self.viz_chart_frame = ScrolledFrame(parent_tab, autohide=True)
        self.viz_chart_frame.pack(fill=BOTH, expand=True, pady=10)

    def setup_training_tab(self, parent_tab):
        control_frame = ttk.Frame(parent_tab)
        control_frame.pack(fill=X, pady=5)
        control_frame.columnconfigure((0, 1), weight=1)

        ttk.Button(control_frame, text="Tune Hyperparameters (Optional, Slow)", command=self.start_tuning_thread,
                   bootstyle="warning").grid(row=0, column=0, sticky=tk.EW, padx=(0, 5), pady=5)
        ttk.Button(control_frame, text="Start Data Preprocessing & Model Training", command=self.start_training_thread,
                   bootstyle="success").grid(row=0, column=1, sticky=tk.EW, padx=(5, 0), pady=5)

        self.progress_bar = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=5)

        self.results_text = tk.Text(parent_tab, height=30, width=100, font=("Courier New", 10), wrap="word")
        self.results_text.pack(pady=10, fill=BOTH, expand=True)
        self.results_text.insert(END, "Training results will be displayed here.")
        self.results_text.config(state=DISABLED)

    def setup_comparison_tab(self, parent_tab):
        plot_controls_frame = ttk.Frame(parent_tab)
        plot_controls_frame.pack(fill=X, pady=5)
        plot_controls_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self.model_selector = ttk.Combobox(plot_controls_frame, state="readonly", bootstyle="info")
        self.model_selector.grid(row=0, column=0, columnspan=4, sticky=tk.EW, padx=(0, 5), pady=(0, 5))

        buttons = {
            "Overall Comparison": self.generate_overall_comparison_chart,
            "Detailed Report": self.generate_detailed_chart,
            "Actual vs. Predicted": self.generate_actual_vs_predicted_chart,
            "Confusion Matrix": self.generate_confusion_matrix_chart
        }
        for i, (text, command) in enumerate(buttons.items()):
            ttk.Button(plot_controls_frame, text=text, command=command, bootstyle="info").grid(row=1, column=i,
                                                                                               sticky=tk.EW, pady=5,
                                                                                               padx=5 if i > 0 else (0,
                                                                                                                     5))

        self.chart_frame = ScrolledFrame(parent_tab, autohide=True)
        self.chart_frame.pack(fill=BOTH, expand=True, pady=10)

    def setup_prediction_tab(self, parent_tab):
        self.prediction_frame = ScrolledFrame(parent_tab, autohide=True)
        self.prediction_frame.pack(fill=BOTH, expand=True, pady=20, padx=20)
        self.prediction_placeholder = ttk.Label(self.prediction_frame,
                                                text="Train a model in Tab 3 to enable prediction.",
                                                font=("Helvetica", 14))
        self.prediction_placeholder.pack()

    def populate_prediction_tab(self):
        for widget in self.prediction_frame.winfo_children(): widget.destroy()
        self.feature_info = data_handler.get_feature_info()
        self.sliders = {};
        self.entries = {}

        input_frame = ttk.Frame(self.prediction_frame)
        input_frame.pack(fill=X, expand=True, pady=10)

        for i, (feature, info) in enumerate(self.feature_info.items()):
            ttk.Label(input_frame, text=f"{feature.replace('_', ' ').title()}", font=("Helvetica", 13)).grid(row=i,
                                                                                                             column=0,
                                                                                                             sticky=tk.W,
                                                                                                             pady=10)

            slider = ttk.Scale(input_frame, from_=info['min'], to=info['max'], orient=HORIZONTAL)
            slider.set(info['default'])
            slider.grid(row=i, column=1, sticky=tk.EW, padx=10)
            self.sliders[feature] = slider

            entry_var = tk.StringVar()
            entry = ttk.Entry(input_frame, textvariable=entry_var, width=10, font=("Helvetica", 13))
            entry.grid(row=i, column=2, sticky=tk.W, padx=5)
            self.entries[feature] = entry_var

            slider.config(command=lambda val, f=feature: self.update_entry_from_slider(f, val))
            entry.bind("<KeyRelease>", lambda event, f=feature: self.update_slider_from_entry(f, event))
            self.update_entry_from_slider(feature, info['default'])

        input_frame.columnconfigure(1, weight=1)

        ttk.Button(input_frame, text="Predict Quality", command=self.predict_quality, bootstyle="success-outline",
                   padding=15).grid(row=len(self.feature_info), column=0, columnspan=3, pady=30, sticky=tk.EW)

        results_frame = ttk.LabelFrame(input_frame, text="Prediction Result", padding="20", bootstyle=INFO)
        results_frame.grid(row=len(self.feature_info) + 1, column=0, columnspan=3, pady=10, sticky=tk.EW)
        results_frame.columnconfigure(0, weight=1)

        self.result_label = ttk.Label(results_frame, text="--", font=("Helvetica", 36, "bold"))
        self.result_label.pack(pady=10)
        self.confidence_label = ttk.Label(results_frame, text="", font=("Helvetica", 16))
        self.confidence_label.pack(pady=5)

    def update_entry_from_slider(self, feature, value):
        val = float(value)
        self.entries[feature].set(f"{val:.4f}" if feature in ['chlorides', 'density'] else f"{val:.2f}")

    def update_slider_from_entry(self, feature, event):
        try:
            val = float(self.entries[feature].get())
            info = self.feature_info[feature]
            if info['min'] <= val <= info['max']: self.sliders[feature].set(val)
        except (ValueError, tk.TclError):
            pass

    # --- Backend Logic & Threading ---
    def load_data(self):
        filepath = filedialog.askopenfilename(title="Select the winequality-red.csv file",
                                              filetypes=[("CSV Files", "*.csv")])
        if not filepath: return
        self.raw_df, info_str = data_handler.load_and_describe_data(filepath)
        if self.raw_df is not None:
            self.data_info_text.config(state=NORMAL);
            self.data_info_text.delete(1.0, END);
            self.data_info_text.insert(END, info_str);
            self.data_info_text.config(state=DISABLED)
            messagebox.showinfo("Success",
                                "Data loaded. You can now explore visualizations in the 'Data Visualization' tab.")
        else:
            messagebox.showerror("Error", info_str)

    def start_training_thread(self):
        if self.raw_df is None: messagebox.showwarning("No Data", "Please load the dataset first."); return
        threading.Thread(target=self.run_training_pipeline, daemon=True).start()

    def start_tuning_thread(self):
        if self.raw_df is None: messagebox.showwarning("No Data", "Please load the dataset first."); return
        threading.Thread(target=self.run_tuning_pipeline, daemon=True).start()

    def run_training_pipeline(self):
        self.progress_bar.start()
        self.update_results("--- Starting Data Preprocessing & Model Training ---\n")
        try:
            self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = data_handler.preprocess_data(
                self.raw_df, self.update_results)
            self.trained_models, self.model_metrics, self.model_reports, self.model_predictions = model_trainer.train_models(
                self.X_train, self.y_train, self.X_test, self.y_test, self.best_params, self.update_results)
            self.model_selector['values'] = list(self.trained_models.keys())
            if self.trained_models: self.model_selector.set(list(self.trained_models.keys())[0])
            self.populate_prediction_tab()
            messagebox.showinfo("Success", "All models trained successfully!")
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred: {e}")
        finally:
            self.progress_bar.stop()

    def run_tuning_pipeline(self):
        self.progress_bar.start()
        self.update_results("--- Starting Hyperparameter Tuning ---\n")
        try:
            X_train_tuned, y_train_tuned = data_handler.preprocess_for_tuning(self.raw_df, self.update_results)
            self.best_params = model_trainer.tune_models(X_train_tuned, y_train_tuned, self.update_results)
            messagebox.showinfo("Tuning Complete",
                                "Hyperparameter tuning finished. The best parameters have been saved and will be used in the next training run.")
        except Exception as e:
            messagebox.showerror("Tuning Error", f"An error occurred: {e}")
        finally:
            self.progress_bar.stop()

    def update_results(self, text):
        self.results_text.config(state=NORMAL);
        self.results_text.insert(END, text);
        self.results_text.see(END);
        self.results_text.config(state=DISABLED)

    # --- Chart Generation ---
    def generate_correlation_heatmap(self):
        visualizer.generate_correlation_heatmap(self.raw_df, self.viz_chart_frame)

    def generate_quality_distributions(self):
        visualizer.generate_quality_distributions(self.raw_df, self.viz_chart_frame)

    def generate_feature_boxplots(self):
        visualizer.generate_feature_boxplots(self.raw_df, self.viz_chart_frame)

    def generate_overall_comparison_chart(self):
        visualizer.generate_overall_comparison_chart(self.model_metrics, self.chart_frame)

    def generate_detailed_chart(self):
        visualizer.generate_detailed_chart(self.model_selector.get(), self.model_reports, self.chart_frame)

    def generate_actual_vs_predicted_chart(self):
        visualizer.generate_actual_vs_predicted_chart(self.model_selector.get(), self.y_test, self.model_predictions,
                                                      self.chart_frame)

    def generate_confusion_matrix_chart(self):
        visualizer.generate_confusion_matrix_chart(self.model_selector.get(), self.y_test, self.model_predictions,
                                                   self.chart_frame)

    # --- Prediction Logic ---
    def predict_quality(self):
        best_model_name = "Random Forest"
        if self.model_metrics:
            best_model_name = max(self.model_metrics, key=lambda k: self.model_metrics[k]['F1-Score (Good)'])

        if best_model_name not in self.trained_models:
            messagebox.showwarning("Model Not Ready", f"The best model ({best_model_name}) is not trained yet.");
            return

        input_data, error_messages = [], []
        for feature, entry_var in self.entries.items():
            value_str = entry_var.get()
            info = self.feature_info[feature]
            try:
                value_float = float(value_str)
                if not (info['min'] <= value_float <= info['max']):
                    error_messages.append(
                        f"- {feature.title()}: Value must be between {info['min']} and {info['max']}.")
                input_data.append(value_float)
            except ValueError:
                error_messages.append(f"- {feature.title()}: Must be a valid number.")

        if error_messages:
            messagebox.showerror("Invalid Input",
                                 "Please correct the following errors:\n\n" + "\n".join(error_messages));
            return

        try:
            input_df = pd.DataFrame([input_data], columns=self.entries.keys())
            scaled_data = self.scaler.transform(input_df)
            model = self.trained_models[best_model_name]
            prediction_numeric = model.predict(scaled_data)[0]
            prediction_proba = model.predict_proba(scaled_data)[0]

            result_text = {0: 'BAD', 1: 'GOOD'}.get(prediction_numeric, 'UNKNOWN')
            confidence = prediction_proba[prediction_numeric] * 100

            self.result_label.config(text=result_text, bootstyle=SUCCESS if result_text == 'GOOD' else DANGER)
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}% (using {best_model_name})")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred during prediction: {e}")


# --- Run the Application ---
if __name__ == "__main__":
    app = MachineLearningGUI()
    app.mainloop()

