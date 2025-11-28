import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('fraud_detection_pipeline.pkl')

# App title and instructions
st.title('Fraud Detection Predection App')
st.markdown('Enter the transaction details below and click Predict.')

# Input fields (matching the screenshot)
transaction_type = st.selectbox('Transaction Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
amount = st.number_input('Amount', min_value=0.0, value=0.0)
old_sender = st.number_input('Old Balance (Sender)', min_value=0.0, value=0.0)
new_sender = st.number_input('New Balance (Sender)', min_value=0.0, value=0.0)
old_receiver = st.number_input('Old Balance (Receiver)', min_value=0.0, value=0.0)
new_receiver = st.number_input('New Balance (Receiver)', min_value=0.0, value=0.0)

# Predict button
if st.button('Predict'):
    # Create input DataFrame with column names matching training
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [old_sender],
        'newbalanceOrig': [new_sender],
        'oldbalanceDest': [old_receiver],
        'newbalanceDest': [new_receiver]
    })
    
    # Add engineered features (balance differences)
    input_data['diff_orig'] = input_data['oldbalanceOrg'] - input_data['newbalanceOrig']
    input_data['diff_dest'] = input_data['newbalanceDest'] - input_data['oldbalanceDest']
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction (as "0" or "1" to match screenshot style, where 0=not fraud, 1=fraud)
    st.subheader(f'Prediction: {prediction}')
    if prediction == 1:
        st.error('This transaction is predicted as FRAUD.')
    else:
        st.success('This transaction is predicted as NOT FRAUD.')