# Imports
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Vinicius Rubens Churn Forescast", page_icon=":100:", layout="centered")

# Load model and scaler
try:
    model = joblib.load('modelling/final_model/model.pkl')
    scaler = joblib.load('pre_processing/artifacts/scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model or Scaler files not found. Please check the file paths.")
    st.stop()

# Function to preprocess input data
# Column names inside the DataFrame must match EXACTLY those used during training
def preprocess_input(
    age, 
    monthly_usage, 
    customer_satisfaction, 
    monthly_value, 
    basic_plan, 
    premium_plan, 
    standard_plan, 
    short_contract, 
    medium_contract,
    long_contract
):
    
    # DataFrame creation
    # scaler/model expects these specific feature names.
    data = pd.DataFrame({
        'Age': [age],
        'MonthlyUsage': [monthly_usage],
        'CustomerSatisfaction': [customer_satisfaction],
        'MonthlyValue': [monthly_value],
        'Plan_Basic': [basic_plan],
        'Plan_Premium': [premium_plan],
        'Plan_Standard': [standard_plan],
        'ContractTime_Long': [long_contract],
        'ContractTime_Medium': [medium_contract],
        'ContractTime_Short': [short_contract]
    })

    # List of columns for scaling
    numeric_cols = [
        'Age', 
        'MonthlyUsage', 
        'CustomerSatisfaction', 
        'MonthlyValue', 
        'Plan_Basic', 
        'Plan_Premium', 
        'Plan_Standard', 
        'ContractTime_Long', 
        'ContractTime_Medium', 
        'ContractTime_Short'
    ]

    # Applying standardization
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    return data

# Function to make predictions
def predict(data):
    prediction = model.predict(data)
    return prediction

# Streamlit Interface
st.title("Churn Predictor with RandomForest")

# Creating input fields
age = st.number_input('Age', min_value=18, max_value=100, value=30)
monthly_usage = st.number_input('Monthly Usage', min_value=0, max_value=200, value=50)
customer_satisfaction = st.number_input('Customer Satisfaction (1-5)', min_value=1, max_value=5, value=3)
monthly_value = st.number_input('Monthly Bill Value', min_value=0.0, max_value=500.0, value=100.0)

# Dropdowns translated
plan = st.selectbox('Plan', ['Basic', 'Premium', 'Standard'])
contract_duration = st.selectbox('Contract Duration', ['Short', 'Medium', 'Long'])

# Button to perform prediction
if st.button('Predict Churn'):

    # One-Hot Encoding Logic
    
    # Plan Logic
    basic_plan = 1 if plan == 'Basic' else 0
    premium_plan = 1 if plan == 'Premium' else 0
    standard_plan = 1 if plan == 'Standard' else 0

    # Contract Duration Logic
    short_contract = 1 if contract_duration == 'Short' else 0
    medium_contract = 1 if contract_duration == 'Medium' else 0
    long_contract = 1 if contract_duration == 'Long' else 0

    # Execute preprocessing
    input_data = preprocess_input(
        age, 
        monthly_usage, 
        customer_satisfaction, 
        monthly_value, 
        basic_plan, 
        premium_plan, 
        standard_plan, 
        short_contract, 
        medium_contract, 
        long_contract
    )

    # Make prediction
    try:
        prediction = predict(input_data)
        
        # Display Result
        result_text = 'Yes' if prediction[0] == 1 else 'No'
        
        if result_text == 'Yes':
            st.error(f'Predicted Churn: {result_text}') # Red highlight for Churn
        else:
            st.success(f'Predicted Churn: {result_text}') # Green highlight for No Churn
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
    
    st.write('Thank you - Vinicius Rubens')