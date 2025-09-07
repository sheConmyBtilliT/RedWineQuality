# --- 1. Import Necessary Libraries ---
import streamlit as st
import pandas as pd
import joblib


# --- 2. Load the Pre-trained Model, Scaler, and Data for UI ---
# We use a caching decorator to ensure this function runs only once.
@st.cache_resource
def load_artifacts():
    """
    Loads the saved RandomForest model, the StandardScaler, and the
    original wine dataset (for UI slider ranges).
    """
    try:
        # Load the pre-trained model
        model = joblib.load('random_forest_model.joblib')

        # Load the fitted scaler
        scaler = joblib.load('scaler.joblib')

        # Load the original dataset to get column names and ranges for sliders
        wine_df = pd.read_csv('winequality-red.csv')

        return model, scaler, wine_df
    except FileNotFoundError:
        st.error("Error: Model or scaler files not found.")
        st.error(
            "Please run the 'red_wine_quality_analysis.py' script first to generate 'random_forest_model.joblib' and 'scaler.joblib'.")
        return None, None, None


# Load the necessary objects
model, scaler, wine_df = load_artifacts()

# --- 3. Streamlit User Interface ---

# Set the title and a descriptive subtitle for the app
st.title('🍷 Red Wine Quality Predictor')
st.markdown("""
This application uses a pre-trained Random Forest model to predict whether a red wine is of 'Good Quality' (score 6 or higher) or 'Bad Quality' (score below 6).

**Use the sliders in the sidebar to input the wine's features.**
""")

# Only build the rest of the UI if the artifacts were loaded successfully
if model and scaler is not None and wine_df is not None:
    # --- 4. Sidebar for User Input ---
    st.sidebar.header('Input Wine Features')


    def user_input_features():
        """
        Creates sliders in the sidebar for each feature and returns a DataFrame
        with the user's selected values.
        """
        # Use the descriptive statistics from the dataframe to set slider min, max, and default values
        desc = wine_df.describe()

        fixed_acidity = st.sidebar.slider('Fixed Acidity', float(desc['fixed acidity']['min']),
                                          float(desc['fixed acidity']['max']), float(desc['fixed acidity']['mean']))
        volatile_acidity = st.sidebar.slider('Volatile Acidity', float(desc['volatile acidity']['min']),
                                             float(desc['volatile acidity']['max']),
                                             float(desc['volatile acidity']['mean']))
        citric_acid = st.sidebar.slider('Citric Acid', float(desc['citric acid']['min']),
                                        float(desc['citric acid']['max']), float(desc['citric acid']['mean']))
        residual_sugar = st.sidebar.slider('Residual Sugar', float(desc['residual sugar']['min']),
                                           float(desc['residual sugar']['max']), float(desc['residual sugar']['mean']))
        chlorides = st.sidebar.slider('Chlorides', float(desc['chlorides']['min']), float(desc['chlorides']['max']),
                                      float(desc['chlorides']['mean']))
        free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', float(desc['free sulfur dioxide']['min']),
                                                float(desc['free sulfur dioxide']['max']),
                                                float(desc['free sulfur dioxide']['mean']))
        total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', float(desc['total sulfur dioxide']['min']),
                                                 float(desc['total sulfur dioxide']['max']),
                                                 float(desc['total sulfur dioxide']['mean']))
        density = st.sidebar.slider('Density', float(desc['density']['min']), float(desc['density']['max']),
                                    float(desc['density']['mean']))
        ph = st.sidebar.slider('pH', float(desc['pH']['min']), float(desc['pH']['max']), float(desc['pH']['mean']))
        sulphates = st.sidebar.slider('Sulphates', float(desc['sulphates']['min']), float(desc['sulphates']['max']),
                                      float(desc['sulphates']['mean']))
        alcohol = st.sidebar.slider('Alcohol', float(desc['alcohol']['min']), float(desc['alcohol']['max']),
                                    float(desc['alcohol']['mean']))

        # Store the inputs in a dictionary using the original feature names
        data = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': ph,
            'sulphates': sulphates,
            'alcohol': alcohol
        }
        # Convert the dictionary to a pandas DataFrame
        features = pd.DataFrame(data, index=[0])
        return features


    # Get user input
    input_df = user_input_features()

    # --- 5. Display User Input and Make Predictions ---

    # Display the user's selected features in the main panel
    st.subheader('Your Selected Wine Features:')
    st.write(input_df)

    # Prediction button
    if st.button('Predict Quality'):
        # Scale the user's input features
        input_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Display the result
        st.subheader('Prediction Result')
        quality = 'Good Quality' if prediction[0] == 1 else 'Bad Quality'

        if quality == 'Good Quality':
            st.success(f'The model predicts: **{quality}** 👍')
        else:
            st.error(f'The model predicts: **{quality}** 👎')

        st.subheader('Prediction Probability')
        # Create a DataFrame for better display of probabilities
        proba_df = pd.DataFrame({
            'Bad Quality': [f"{prediction_proba[0][0]:.2%}"],
            'Good Quality': [f"{prediction_proba[0][1]:.2%}"]
        })
        st.write(proba_df)