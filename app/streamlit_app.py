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

# ---------------- CSS (AUTO LIGHT + DARK) ----------------
st.markdown("""
<style>
/* Cards adapt automatically */
.card {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.2);
    background-color: rgba(128,128,128,0.05);
    margin-bottom: 1rem;
}

/* Metric boxes */
.metric-box {
    padding: 1rem;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(128,128,128,0.2);
    background-color: rgba(128,128,128,0.08);
}

/* Titles */
.main-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
}

.sub-text {
    text-align: center;
    opacity: 0.7;
    margin-bottom: 1.5rem;
}

/* Small text */
.small-note {
    text-align: center;
    opacity: 0.6;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🛡️ Insurance Claim Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict fraud risk using machine learning</div>', unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📋 Claim Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    months = st.number_input("Months as Customer", 0, 500, 12)
    premium = st.number_input("Annual Premium", 100, 20000, 5000)
    total_claim = st.number_input("Total Claim Amount", 0, 100000, 20000)
    vehicle_claim = st.number_input("Vehicle Claim", 0, 100000, 15000)

with col2:
    witnesses = st.number_input("Witnesses", 0, 5, 1)
    injuries = st.number_input("Bodily Injuries", 0, 5, 0)
    incident_hour = st.slider("Incident Hour", 0, 23, 12)
    vehicles = st.number_input("Vehicles Involved", 1, 5, 1)
    threshold = st.slider("Risk Threshold", 0.10, 0.60, 0.30, 0.01)

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

    # Default encoded values (important for trained model)
    defaults = [
        "policy_state_OH",
        "policy_csl_250/500",
        "insured_sex_MALE",
        "incident_type_Single Vehicle Collision",
        "collision_type_Front Collision",
        "incident_severity_Total Loss",
        "authorities_contacted_Police",
        "property_damage_YES",
        "police_report_available_YES",
    ]

    for col in defaults:
        if col in input_df.columns:
            input_df.at[0, col] = 1

    probability = model.predict_proba(input_df)[0][1]
    risk_pct = probability * 100

    st.subheader("📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.markdown(
            f'<div class="metric-box"><h4>Fraud Probability</h4><h2>{risk_pct:.2f}%</h2></div>',
            unsafe_allow_html=True
        )

    with colB:
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
        f'<div class="small-note">Threshold: {threshold:.2f} — lower = more sensitive to fraud</div>',
        unsafe_allow_html=True
    )

    # ---------------- SUMMARY ----------------
    st.subheader("📄 Input Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "Age", "Months", "Premium", "Total Claim",
            "Vehicle Claim", "Witnesses", "Injuries",
            "Hour", "Vehicles"
        ],
        "Value": [
            age, months, premium, total_claim,
            vehicle_claim, witnesses, injuries,
            incident_hour, vehicles
        ]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Demo tool — not for real insurance decisions.")