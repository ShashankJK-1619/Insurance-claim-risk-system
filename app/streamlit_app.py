import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Claim Risk Prediction System", layout="centered")

st.title("Insurance Claim Risk Prediction System")
st.write("Enter customer and claim details to estimate fraud/claim risk.")

model = joblib.load("models/claim_model.pkl")

months_as_customer = st.number_input("Months as Customer", min_value=0, value=120)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
policy_annual_premium = st.number_input("Policy Annual Premium", min_value=0.0, value=1200.0)

if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        "months_as_customer": months_as_customer,
        "age": age,
        "policy_annual_premium": policy_annual_premium
    }])

    st.write(input_df)