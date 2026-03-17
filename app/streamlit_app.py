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
        "policy_state": 0,
        "policy_csl": 0,
        "policy_deductable": 500,
        "policy_annual_premium": policy_annual_premium,
        "umbrella_limit": 0,
        "insured_zip": 0,
        "insured_sex": 0,
        "insured_education_level": 0,
        "insured_occupation": 0,
        "insured_hobbies": 0,
        "insured_relationship": 0,
        "capital-gains": 0,
        "capital-loss": 0,
        "incident_type": 0,
        "collision_type": 0,
        "incident_severity": 0,
        "authorities_contacted": 0,
        "incident_state": 0,
        "incident_city": 0,
        "incident_location": 0,
        "incident_hour_of_the_day": 12,
        "number_of_vehicles_involved": 1,
        "property_damage": 0,
        "bodily_injuries": 0,
        "witnesses": 0,
        "police_report_available": 0,
        "total_claim_amount": 5000,
        "injury_claim": 0,
        "property_claim": 0,
        "vehicle_claim": 5000,
        "auto_make": 0,
        "auto_model": 0,
        "auto_year": 2015,
        "policy_bind_date_year": 2014,
        "policy_bind_date_month": 1,
        "policy_bind_date_day": 1,
        "incident_date_year": 2015,
        "incident_date_month": 1,
        "incident_date_day": 1
    }])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ High Risk Claim (Potential Fraud)")
    else:
        st.success("✅ Low Risk Claim")