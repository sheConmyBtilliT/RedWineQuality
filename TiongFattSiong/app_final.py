import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import joblib
import pandas as pd
import numpy as np

# --- Load the Best Model (SVM) and Scaler ---
try:
    scaler = joblib.load('scaler.joblib')
    svm_model = joblib.load('svm_model.joblib')
except FileNotFoundError:
    messagebox.showerror("Error", "Model files not found! Please run 'train_best_model.py' first.")
    exit()

# --- Feature names and reasonable ranges for UI sliders ---
feature_info = {
    'fixed acidity': {'min': 4.6, 'max': 15.9, 'default': 7.4},
    'volatile acidity': {'min': 0.1, 'max': 1.6, 'default': 0.7},
    'citric acid': {'min': 0, 'max': 1, 'default': 0.0},
    'residual sugar': {'min': 0.9, 'max': 16, 'default': 1.9},
    'chlorides': {'min': 0.01, 'max': 0.6, 'default': 0.076},
    'free sulfur dioxide': {'min': 1, 'max': 72, 'default': 11.0},
    'total sulfur dioxide': {'min': 6, 'max': 289, 'default': 34.0},
    'density': {'min': 0.990, 'max': 1.004, 'default': 0.9978},
    'pH': {'min': 2.7, 'max': 4.0, 'default': 3.51},
    'sulphates': {'min': 0.3, 'max': 2.0, 'default': 0.56},
    'alcohol': {'min': 8, 'max': 15, 'default': 9.4}
}


# --- UI Application Class ---
class SvmPredictorApp(ttk.Window):
    def __init__(self):
        # Use the 'superhero' theme from ttkbootstrap for a modern, dark look
        super().__init__(themename="superhero")
        self.title("Wine Quality Predictor - SVM")
        self.geometry("600x900")
        self.resizable(False, False)

        self.sliders = {}
        self.value_labels = {}
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main_frame, text="Wine Quality Predictor", font=("Helvetica", 24, "bold"), bootstyle=PRIMARY)
        header.pack(pady=(0, 20))

        # Input Frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=X, expand=True, pady=10)

        # Create input sliders dynamically
        for i, (feature, info) in enumerate(feature_info.items()):
            # Feature Label
            label = ttk.Label(input_frame, text=f"{feature.replace('_', ' ').title()}", font=("Helvetica", 11))
            label.grid(row=i, column=0, sticky=W, pady=8)

            # Slider
            slider = ttk.Scale(input_frame, from_=info['min'], to=info['max'], orient=HORIZONTAL, length=250)
            slider.set(info['default'])
            slider.grid(row=i, column=1, sticky=EW, padx=10)
            self.sliders[feature] = slider

            # Value Label (shows the slider's current value)
            value_label = ttk.Label(input_frame, text=f"{info['default']:.2f}", font=("Helvetica", 11, "bold"))
            value_label.grid(row=i, column=2, sticky=W, padx=5)
            self.value_labels[feature] = value_label

            # Link slider movement to update the value label
            slider.config(command=lambda val, f=feature: self.update_label(f, val))

        input_frame.columnconfigure(1, weight=1)

        # Predict Button
        predict_button = ttk.Button(main_frame, text="Predict Quality", command=self.predict_quality,
                                    bootstyle="success-outline", padding=15)
        predict_button.pack(pady=25, fill=X)

        # Results Frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding="20", bootstyle=INFO)
        results_frame.pack(fill=X, expand=True)
        results_frame.columnconfigure(0, weight=1)

        self.result_label = ttk.Label(results_frame, text="--", font=("Helvetica", 32, "bold"))
        self.result_label.pack(pady=10)
        self.confidence_label = ttk.Label(results_frame, text="", font=("Helvetica", 14))
        self.confidence_label.pack(pady=5)

    def update_label(self, feature, value):
        if feature in ['chlorides', 'density']:
            formatted_value = f"{float(value):.4f}"
        else:
            formatted_value = f"{float(value):.2f}"
        self.value_labels[feature].config(text=formatted_value)

    def predict_quality(self):
        try:
            # Collect data from sliders
            input_data = [self.sliders[feature].get() for feature in feature_info]

            input_df = pd.DataFrame([input_data], columns=feature_info.keys())
            scaled_data = scaler.transform(input_df)
            scaled_data_df = pd.DataFrame(scaled_data, columns=feature_info.keys())

            # Make prediction and get probabilities
            prediction_numeric = svm_model.predict(scaled_data_df)[0]
            prediction_proba = svm_model.predict_proba(scaled_data_df)[0]

            # Map numeric prediction to string label
            class_mapping = {0: 'BAD', 1: 'GOOD'}
            result_text = class_mapping.get(prediction_numeric, 'UNKNOWN')

            # Get the confidence score for the predicted class
            confidence = prediction_proba[prediction_numeric] * 100

            # Update the UI with the result and confidence
            self.result_label.config(text=result_text)
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

            # Change color based on result
            if result_text == 'GOOD':
                self.result_label.config(bootstyle=SUCCESS)
            else:
                self.result_label.config(bootstyle=DANGER)

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred: {e}")


# --- Run the Application ---
if __name__ == "__main__":
    app = SvmPredictorApp()
    app.mainloop()