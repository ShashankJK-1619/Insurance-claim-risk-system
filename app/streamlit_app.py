import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Claim Risk Prediction System", layout="centered")

st.title("Insurance Claim Risk Prediction System")
st.write("Enter customer and claim details to estimate fraud/claim risk.")

model = joblib.load("models/claim_model.pkl")
feature_names = list(model.feature_names_in_)

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
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)

    numeric_values = {
        "months_as_customer": months_as_customer,
        "age": age,
        "policy_deductable": 500,
        "policy_annual_premium": policy_annual_premium,
        "umbrella_limit": 0,
        "capital-gains": 0,
        "capital-loss": 0,
        "incident_hour_of_the_day": incident_hour_of_the_day,
        "number_of_vehicles_involved": number_of_vehicles_involved,
        "bodily_injuries": bodily_injuries,
        "witnesses": witnesses,
        "total_claim_amount": total_claim_amount,
        "injury_claim": 10000,
        "property_claim": 10000,
        "vehicle_claim": vehicle_claim,
        "auto_year": 2015,
    }

    for col, val in numeric_values.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    fraud_like_defaults = [
        "policy_state_OH",
        "policy_csl_250/500",
        "insured_sex_MALE",
        "insured_education_level_High School",
        "insured_occupation_craft-repair",
        "insured_hobbies_cross-fit",
        "insured_relationship_own-child",
        "incident_type_Single Vehicle Collision",
        "collision_type_Front Collision",
        "incident_severity_Total Loss",
        "authorities_contacted_Police",
        "incident_state_SC",
        "incident_city_Northbend",
        "property_damage_YES",
        "police_report_available_YES",
        "auto_make_Honda",
        "auto_model_Civic",
    ]

    for col in fraud_like_defaults:
        if col in input_df.columns:
            input_df.at[0, col] = 1

    probability = model.predict_proba(input_df)[0][1]

    st.write(f"Fraud Probability: {probability:.2%}")

    if probability > 0.30:
        st.error("⚠️ High Risk Claim (Potential Fraud)")
    else:
        st.success("✅ Low Risk Claim")