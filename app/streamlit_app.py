import streamlit as st
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Insurance Claim Risk Prediction System",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/claim_model.pkl")
model_features = list(model.feature_names_in_)

# ---------------- STYLES ----------------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        text-align: center;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .metric-box {
        padding: 1rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🛡️ Insurance Claim Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Estimate whether a claim appears low risk or potentially fraudulent using a trained machine learning model.</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Claim Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    months = st.number_input("Months as Customer", min_value=0, max_value=500, value=12)
    premium = st.number_input("Annual Premium", min_value=100, max_value=20000, value=5000)
    total_claim = st.number_input("Total Claim Amount", min_value=0, max_value=100000, value=20000)
    vehicle_claim = st.number_input("Vehicle Claim", min_value=0, max_value=100000, value=15000)

with col2:
    witnesses = st.number_input("Witnesses", min_value=0, max_value=5, value=1)
    injuries = st.number_input("Bodily Injuries", min_value=0, max_value=5, value=0)
    incident_hour = st.slider("Incident Hour", min_value=0, max_value=23, value=12)
    vehicles = st.number_input("Vehicles Involved", min_value=1, max_value=5, value=1)
    threshold = st.slider("Risk Threshold", min_value=0.10, max_value=0.60, value=0.30, step=0.01)

predict_clicked = st.button("🔍 Predict Risk", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_clicked:
    input_df = pd.DataFrame(0, index=[0], columns=model_features)

    numeric_values = {
        "age": age,
        "months_as_customer": months,
        "policy_annual_premium": premium,
        "total_claim_amount": total_claim,
        "vehicle_claim": vehicle_claim,
        "witnesses": witnesses,
        "bodily_injuries": injuries,
        "incident_hour_of_the_day": incident_hour,
        "number_of_vehicles_involved": vehicles,
        "policy_deductable": 500,
        "umbrella_limit": 0,
        "capital-gains": 0,
        "capital-loss": 0,
        "injury_claim": min(total_claim * 0.2, 20000),
        "property_claim": min(total_claim * 0.2, 20000),
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
    risk_pct = probability * 100

    st.subheader("Prediction Result")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f'<div class="metric-box"><h4>Fraud Probability</h4><h2>{risk_pct:.2f}%</h2></div>',
            unsafe_allow_html=True
        )
    with m2:
        label = "High Risk" if probability > threshold else "Low Risk"
        st.markdown(
            f'<div class="metric-box"><h4>Risk Label</h4><h2>{label}</h2></div>',
            unsafe_allow_html=True
        )

    st.progress(float(probability))

    if probability > threshold:
        st.error("🚨 High Risk Claim (Potential Fraud)")
    else:
        st.success("✅ Low Risk Claim")

    st.markdown(
        f'<div class="small-note">Current decision threshold: {threshold:.2f}. Lower thresholds make the model more sensitive to potential fraud.</div>',
        unsafe_allow_html=True
    )

    st.subheader("Input Summary")
    summary_df = pd.DataFrame({
        "Feature": [
            "Age",
            "Months as Customer",
            "Annual Premium",
            "Total Claim Amount",
            "Vehicle Claim",
            "Witnesses",
            "Bodily Injuries",
            "Incident Hour",
            "Vehicles Involved"
        ],
        "Value": [
            age,
            months,
            premium,
            total_claim,
            vehicle_claim,
            witnesses,
            injuries,
            incident_hour,
            vehicles
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("This tool is for demonstration purposes and should not be used as a sole basis for real insurance decisions.")