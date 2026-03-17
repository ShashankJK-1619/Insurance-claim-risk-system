import streamlit as st

st.title("Insurance Claim Risk Prediction")

st.write("App is running successfully 🚀")

st.write("Test input:")
age = st.slider("Age", 18, 100, 30)

st.write(f"Selected age: {age}")