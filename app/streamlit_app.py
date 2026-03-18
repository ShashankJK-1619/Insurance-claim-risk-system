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
total_claim_amount = st.number_input("Total Claim Amount", min_value=0.0, value=5000.0)
vehicle_claim = st.number_input("Vehicle Claim", min_value=0.0, value=5000.0)
witnesses = st.number_input("Witnesses", min_value=0, value=0)
bodily_injuries = st.number_input("Bodily Injuries", min_value=0, value=0)
incident_hour_of_the_day = st.number_input("Incident Hour of the Day", min_value=0, max_value=23, value=12)
number_of_vehicles_involved = st.number_input("Number of Vehicles Involved", min_value=1, value=1)

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
        "incident_hour_of_the_day": incident_hour_of_the_day,
        "number_of_vehicles_involved": number_of_vehicles_involved,
        "property_damage": 1,
        "bodily_injuries": bodily_injuries,
        "witnesses": witnesses,
        "police_report_available": 1,
        "total_claim_amount": total_claim_amount,
        "injury_claim": 10000,
        "property_claim": 10000,
        "vehicle_claim": vehicle_claim,
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
    probability = model.predict_proba(input_df)[0][1]

    st.write(f"Fraud probability: {probability:.2%}")

    if prediction == 1:
        st.error("⚠️ High Risk Claim (Potential Fraud)")
    else:
        st.success("✅ Low Risk Claim")