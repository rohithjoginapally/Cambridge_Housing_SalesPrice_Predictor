import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('compressed_random_forest_model.joblib')

# Default values for each feature
default_values = {
   'BuildingValue': 684700.0, 
   'LandValue': 0.0, 
   'AssessedValue': 684700.0, 
   'Interior_LivingArea': 980.0, 
   'Interior_TotalRooms': 5.0, 
   'Interior_Bedrooms': 2.0, 
   'Parking_Garage': 0.0, 
   'Years_Since_Sale': 12.0, 
   'Building_Age': 13.0, 
}

# Streamlit app layout
st.title('Cambridge Condominium Sale Price Prediction App - Regression')

# Create input fields for each feature
input_data = {}
for feature, default in default_values.items():
    if isinstance(default, bool):  # For boolean features
        input_data[feature] = st.checkbox(feature, value=default)
    elif isinstance(default, (int, float)):  # For numerical features
        input_data[feature] = st.number_input(feature, value=default)
    else:  # For categorical features
        input_data[feature] = st.selectbox(feature, [default])

# Button to make prediction
if st.button('Predict Sale Price'):
    # Create a DataFrame for prediction
    df_predict = pd.DataFrame([input_data])

    # Make a prediction
    prediction = model.predict(df_predict)

    # Display the prediction
    st.write(f'Predicted Sale Price: ${prediction[0]:,.2f}')