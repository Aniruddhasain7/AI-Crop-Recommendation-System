import streamlit as st
import pandas as pd
import joblib
import base64
import os

pipeline_bundle = joblib.load("crop_model.joblib")
model = pipeline_bundle["model"]
scaler = pipeline_bundle["scaler"]
label_encoder = pipeline_bundle["label_encoder"]

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

*, *::before, *::after {
    box-sizing: border-box;
}

html, body {
    max-width: 100vw;
    overflow-x: hidden !important;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif;
    background-color: #050505 !important;
    color: #FFFFFF;
    max-width: 100vw;
    overflow-x: hidden !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1100px;
    width: 100% !important;
    padding: 2.5rem 1.5rem;
    box-sizing: border-box !important;
}

div[data-testid="stImage"] img {
    border-radius: 20px !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.15) !important;
    margin: 0 auto !important;
    display: block !important;
    max-width: 100% !important;
    height: auto !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
}

.stMarkdown h3 {
    font-family: 'Outfit', sans-serif !important;
    color: #10B981 !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    margin-top: 15px !important;
    margin-bottom: 16px !important;
    border-bottom: 2px solid rgba(16, 185, 129, 0.15) !important;
    padding-bottom: 8px !important;
}

.main-header {
    text-align: center;
    margin-bottom: 35px;
    width: 100%;
    box-sizing: border-box;
}
.main-header h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: clamp(1.6rem, 4.5vw, 2.8rem) !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #10B981 0%, #34D399 50%, #A7F3D0 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-top: 12px !important;
    margin-bottom: 10px !important;
    line-height: 1.2 !important;
    word-break: break-word;
}
.main-header p {
    font-family: 'Inter', sans-serif !important;
    color: #94A3B8 !important;
    font-size: clamp(0.88rem, 2.2vw, 1.05rem) !important;
    max-width: 700px;
    margin: 0 auto !important;
    opacity: 0.9;
    line-height: 1.6;
    word-break: break-word;
}

.logo-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 8px;
    width: 100%;
}
.logo-wrapper img {
    width: 130px;
    max-width: 100%;
    height: auto;
    border-radius: 20px !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.15) !important;
}

/* Streamlit responsive column wrap for tablets and mobile */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 10px !important;
    }
    [data-testid="column"] {
        min-width: calc(50% - 6px) !important;
        flex: 1 1 calc(50% - 6px) !important;
        max-width: 100% !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding: 1.25rem 0.75rem !important;
    }
    .logo-wrapper img {
        width: 85px !important;
    }
    .main-header {
        margin-bottom: 20px !important;
    }
    .crop-result-card {
        padding: 20px 14px !important;
        border-radius: 18px !important;
        margin-top: 20px !important;
    }
    .confidence-text {
        font-size: 0.95rem !important;
        margin-bottom: 12px !important;
    }
    .prob-list-title {
        font-size: 1.1rem !important;
        margin-top: 20px !important;
    }
    .prob-item {
        padding: 10px 12px !important;
    }
    div[data-testid="stNumberInput"] {
        padding: 10px !important;
    }
    div.stButton > button {
        padding: 12px 20px !important;
        font-size: 1.05rem !important;
    }
}

@media (max-width: 480px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 8px !important;
    }
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
        width: 100% !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 0.5rem !important;
    }
    .logo-wrapper img {
        width: 70px !important;
    }
    .crop-result-card {
        padding: 16px 10px !important;
    }
    .prob-info {
        font-size: 0.88rem !important;
    }
}

div[data-testid="stNumberInput"] {
    background-color: rgba(18, 18, 18, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    margin-bottom: 12px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stNumberInput"]:focus-within {
    border-color: rgba(16, 185, 129, 0.5) !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.15) !important;
    background-color: rgba(22, 22, 22, 0.8) !important;
}
div[data-baseweb="input"] {
    background-color: rgba(10, 10, 10, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease;
    width: 100% !important;
    box-sizing: border-box !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #10B981 !important;
    box-shadow: none !important;
}
input {
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
}

label[data-testid="stWidgetLabel"] p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    margin-bottom: 6px !important;
    transition: color 0.3s ease;
}
div[data-testid="stNumberInput"]:focus-within label p {
    color: #34D399 !important;
}

div[data-testid="stNumberInput"] button {
    background-color: rgba(16, 185, 129, 0.08) !important;
    border: none !important;
    color: #34D399 !important;
    border-radius: 6px !important;
}
div[data-testid="stNumberInput"] button:hover {
    background-color: rgba(16, 185, 129, 0.2) !important;
    color: #10B981 !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 14px 28px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
    box-sizing: border-box !important;
    margin-top: 15px;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35) !important;
    background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important;
}
div.stButton > button:active {
    transform: translateY(1px) !important;
}

.crop-result-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(10, 10, 10, 0.9) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 24px;
    padding: 28px 20px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(16, 185, 129, 0.05);
    animation: fadeIn 0.6s ease-out;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    overflow: hidden;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 2px;
    color: #34D399;
}
.crop-badge {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: clamp(1.25rem, 5vw, 2.2rem);
    letter-spacing: 1px;
    padding: 10px 20px;
    border-radius: 12px;
    display: inline-block;
    margin: 14px 0;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    text-transform: uppercase;
    max-width: 100%;
    box-sizing: border-box;
    word-break: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
    line-height: 1.2;
}
.confidence-text {
    font-size: 1.05rem;
    color: #CBD5E1;
    margin-bottom: 14px;
    word-break: break-word;
}
.confidence-text strong {
    color: #10B981;
}
.progress-bar-outer {
    background-color: rgba(0, 0, 0, 0.6);
    border-radius: 10px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-sizing: border-box;
}
.progress-bar-inner {
    background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
    height: 100%;
    border-radius: 10px;
    max-width: 100%;
}
.prob-list-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-top: 30px;
    margin-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 8px;
    width: 100%;
    box-sizing: border-box;
}
.prob-item {
    background: rgba(18, 18, 18, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    width: 100%;
    box-sizing: border-box;
}
.prob-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    gap: 8px;
}
.prob-name {
    color: #E2E8F0;
    text-transform: capitalize;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.prob-percentage {
    color: #34D399;
    flex-shrink: 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

logo_path = os.path.join("assets", "crop_logo.png")
with open(logo_path, "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div class="logo-wrapper">
    <img src="data:image/png;base64,{logo_b64}" alt="Crop Recommendation Logo">
</div>
<div class="main-header">
    <h1>Crop Recommendation System</h1>
    <p>Optimize your agricultural yield with machine learning. Enter the soil profiles and climate conditions to receive custom crop recommendations.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🧪 Soil Nutrients")
col_n, col_p, col_k = st.columns(3)
with col_n:
    N = st.number_input("Nitrogen (N)", 0.0, 300.0, step=1.0)
with col_p:
    P = st.number_input("Phosphorus (P)", 0.0, 150.0, step=1.0)
with col_k:
    K = st.number_input("Potassium (K)", 0.0, 200.0, step=1.0)

st.markdown("### 🌦️ Environmental Factors")
col_t, col_h, col_ph, col_r = st.columns(4)
with col_t:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, step=1.0)
with col_h:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, step=1.0)
with col_ph:
    ph = st.number_input("Soil pH", 3.5, 9.0, step=0.1)
with col_r:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 2000.0, step=10.0)

st.write("")

if st.button("Recommend Crop"):
    input_df = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=[
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
    ])

    input_scaled = scaler.transform(input_df)
    pred_encoded = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    crop = label_encoder.inverse_transform([pred_encoded])[0]
    confidence = probs[pred_encoded] * 100

    st.markdown(f"""
        <div class="crop-result-card">
            <div class="result-label">🌱 RECOMMENDED CROP</div>
            <div class="crop-badge">{crop.upper()}</div>
            <div class="confidence-text">
                Recommendation Confidence: <strong>{confidence:.2f}%</strong>
            </div>
            <div class="progress-bar-outer">
                <div class="progress-bar-inner" style="width: {confidence}%"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="prob-list-title">📊 Top Crop Probabilities</div>', unsafe_allow_html=True)

    top_indices = probs.argsort()[-3:][::-1]
    prob_html = ""
    for idx in top_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        prob = probs[idx]
        prob_pct = prob * 100
        prob_html += f"""
        <div class="prob-item">
            <div class="prob-info">
                <span class="prob-name">{crop_name}</span>
                <span class="prob-percentage">{prob_pct:.2f}%</span>
            </div>
            <div class="progress-bar-outer">
                <div class="progress-bar-inner" style="width: {prob_pct}%"></div>
            </div>
        </div>
        """
    st.markdown(prob_html, unsafe_allow_html=True)
