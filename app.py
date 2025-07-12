import streamlit as st
import pandas as pd
import joblib  # or use pickle
import numpy as np

# Load pipeline
loaded_pipeline = joblib.load("logistic_pipeline.joblib")

#title
st.set_page_config(page_title='IPO Application Suggestion')

st.title("IPO Application Suggestion")

# Input form
with st.form("input_form"):
    issue_price = st.number_input("Issue Price", min_value=0.0, step=0.01)
    size_in_cr = st.number_input("Size (in Cr)", min_value=0.0, step=0.01)
    qib = st.number_input("QIB", min_value=0.0, step=0.01)

    submitted = st.form_submit_button("Predict")

# Make prediction
if submitted:
    # Encode gender manually or with OneHotEncoder/LabelEncoder logic
    # gender_encoded = 0 if gender == "Male" else 1

    # Prepare input
    input_data = pd.DataFrame([[issue_price, size_in_cr, qib]], columns=["issue_price", "size_in_cr", "qib"])

    # Predict
    prediction = loaded_pipeline.predict(input_data)[0]
    if prediction == 1:
        st.success("Prediction: Yes")
    else:
        st.error("Prediction: No")