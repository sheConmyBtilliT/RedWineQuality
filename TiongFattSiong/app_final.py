import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

# --- Load the Best Model (SVM) and Scaler ---
try:
    scaler = joblib.load('scaler.joblib')
    svm_model = joblib.load('svm_model.joblib')
except FileNotFoundError:
    messagebox.showerror("Error", "Model files not found! Please run 'train_best_model.py' first to create them.")
    exit()

# --- Feature names in the correct order for prediction ---
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]


# --- UI Application Class ---
class SvmPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wine Quality Predictor (SVM Model)")
        self.geometry("550x800")
        self.resizable(False, False)

        # Style configuration for a modern look
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TLabel', font=('Helvetica', 12))
        self.style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=10)
        self.style.configure('TEntry', font=('Helvetica', 12), padding=5)
        self.style.configure('Header.TLabel', font=('Helvetica', 18, 'bold'), foreground='#333')
        self.style.configure('Good.TLabel', font=('Helvetica', 24, 'bold'), foreground='green')
        self.style.configure('Bad.TLabel', font=('Helvetica', 24, 'bold'), foreground='red')

        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_label = ttk.Label(main_frame, text="Enter Wine Characteristics", style='Header.TLabel')
        header_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Create input fields dynamically
        for i, feature in enumerate(feature_names):
            label = ttk.Label(main_frame, text=f"{feature.replace('_', ' ').title()}:")
            label.grid(row=i + 1, column=0, sticky=tk.W, pady=6, padx=5)

            entry = ttk.Entry(main_frame, width=20)
            entry.grid(row=i + 1, column=1, sticky=tk.EW, pady=6, padx=5)
            self.entries[feature] = entry

        # Predict Button with modern styling
        predict_button = ttk.Button(main_frame, text="Predict Quality", command=self.predict_quality, style='TButton')
        predict_button.grid(row=len(feature_names) + 1, column=0, columnspan=2, pady=25)

        # Results Frame for a clean output display
        results_frame = ttk.LabelFrame(main_frame, text="Predicted Quality", padding="20")
        results_frame.grid(row=len(feature_names) + 2, column=0, columnspan=2, sticky=tk.EW)
        results_frame.columnconfigure(0, weight=1)  # Center the result

        self.result_label = ttk.Label(results_frame, text="--", font=('Helvetica', 24, 'bold'))
        self.result_label.grid(row=0, column=0, pady=10)

    def predict_quality(self):
        try:
            # 1. Collect and validate input data from entry fields
            input_data = [float(self.entries[feature].get()) for feature in feature_names]

            # 2. Create a DataFrame for the scaler (it expects a DataFrame)
            input_df = pd.DataFrame([input_data], columns=feature_names)

            # 3. Scale the data using the loaded scaler
            scaled_data = scaler.transform(input_df)

            scaled_data_df = pd.DataFrame(scaled_data, columns=feature_names)

            # Make prediction using the loaded SVM model
            prediction_numeric = svm_model.predict(scaled_data_df)[0]

            # --- MORE ROBUST FIX APPLIED HERE ---
            # Explicitly map the numeric output (0 or 1) to the string label.
            # Scikit-learn typically assigns numbers alphabetically: 'bad' -> 0, 'good' -> 1
            class_mapping = {0: 'bad', 1: 'good'}
            predicted_class_name = class_mapping.get(prediction_numeric, 'Unknown')

            # Display the result with color coding
            result_text = predicted_class_name.upper()

            self.result_label.config(text=result_text)

            if result_text == 'GOOD':
                self.result_label.config(style='Good.TLabel')
            else:
                self.result_label.config(style='Bad.TLabel')

        except ValueError:
            messagebox.showwarning("Input Error", "Please ensure all fields are filled with valid numbers.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred: {e}")


# --- Run the Application ---
if __name__ == "__main__":
    app = SvmPredictorApp()
    app.mainloop()