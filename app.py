import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st

# Load your trained model
model = joblib.load('model.pkl') 


def predict_loan_approval(data):
    # Encode categorical variables
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Make prediction
    prediction = model.predict(data)

    return prediction[0]


def main():
    st.title("Loan Approval Prediction")
    
    
     # Collect user input
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, step=1)
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input("Applicant's Income")
    coapplicant_income = st.number_input("Coapplicant's Income")
    loan_amount = st.number_input('Loan Amount')
    loan_amount_term = st.number_input('Loan Amount Term (in months)')
    credit_history = st.selectbox('Credit History', [1.0, 0.0])
    property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])
    
    
       # Prepare input data
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    if st.button('Predict'):
        prediction = predict_loan_approval(df)
        if prediction == 1:
            st.success("Congratulations! Your loan is likely to be approved.")
        else:
            st.error("Sorry, your loan is likely to be rejected.")
    
    
if __name__ == '__main__':
    main()