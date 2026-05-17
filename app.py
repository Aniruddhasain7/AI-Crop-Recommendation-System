import streamlit as st
import pandas as pd
import joblib

model = joblib.load("crop_recommendation_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and environmental values to get the best crop suggestion")

N = st.number_input("Nitrogen (N)", 0.0, 300.0, step=1.0)
P = st.number_input("Phosphorus (P)", 0.0, 150.0, step=1.0)
K = st.number_input("Potassium (K)", 0.0, 200.0, step=1.0)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, step=1.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, step=1.0)
ph = st.number_input("Soil pH", 3.5, 9.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", 0.0, 2000.0, step=10.0)

if st.button("Recommend Crop"):

    input_df = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=[
        'N','P','K','temperature','humidity','ph','rainfall'
    ])

    input_scaled = scaler.transform(input_df)

    pred_encoded = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]

    crop = label_encoder.inverse_transform([pred_encoded])[0]
    confidence = probs[pred_encoded] * 100

    st.divider()

    st.success(f"🌱 Recommended Crop: **{crop.upper()}**")
    st.info(f"🔍 Recommendation Confidence: **{confidence:.2f}%**")

    st.subheader("📊 Top Crop Probabilities")

    top_indices = probs.argsort()[-3:][::-1]

    for idx in top_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        prob = probs[idx]
        st.write(f"{crop_name}: {prob*100:.2f}%")
        st.progress(float(prob))
