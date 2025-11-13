import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium
import math
from dotenv import load_dotenv
import os
from tensorflow.keras.models import load_model  # ✅ for Deep Learning model
from sklearn.preprocessing import MinMaxScaler  # ✅ for data scaling

# === Load environment variables ===
load_dotenv()

# === Page Config ===
st.set_page_config(page_title="🏙️ City Traffic Prediction", layout="centered")
st.title("🏙️ City Traffic Prediction (Live Data)")

# === Load Models & Assets ===
try:
    model = joblib.load("best_ml_model.pkl")
    cls_model = joblib.load("best_cls_model.pkl")
    le = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except Exception:
    st.error("❌ Models or feature_columns.pkl not found. Run main.py first to train models.")
    st.stop()

# === Try loading the Deep Learning model ===
try:
    dl_model = load_model("best_dl_model.keras")
    st.sidebar.success("🧠 Deep Learning model loaded successfully!")
except Exception:
    dl_model = None
    st.sidebar.warning("⚠️ Deep Learning model not found. Run src/train_dl.py first.")

# === CSS Styling ===
st.markdown("""
<style>
body {
    background: #f8f9fa;
    color: #333;
}
.main {
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    margin-top: 25px;
}
h1 {
    text-align: center;
    color: #0077b6;
    font-size: 2.3rem;
    margin-bottom: 10px;
}
h2, h3 {
    text-align: center;
    color: #00b4d8;
}
.stButton>button {
    background-color: #00b4d8;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 3em;
    font-weight: 600;
    border: none;
}
.stButton>button:hover { background-color: #0077b6; }
.refresh-button>button {
    background-color: #6c757d;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 3em;
}
.refresh-button>button:hover { background-color: #495057; }
.tip {
    font-size: 1.1rem;
    font-weight: 500;
    text-align: center;
    padding: 10px;
    border-radius: 10px;
    margin-top: 15px;
}
.low-tip { background-color: #d4edda; color: #155724; }
.moderate-tip { background-color: #fff3cd; color: #856404; }
.high-tip { background-color: #f8d7da; color: #721c24; }
.refresh-info {
    text-align: center;
    color: #555;
    margin-top: 10px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# === API Keys ===
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
TOMTOM_KEY = os.getenv("TOMTOM_API_KEY", "")

# === Maintain State ===
if "city" not in st.session_state:
    st.session_state.city = "Bangalore"
if "last_update" not in st.session_state:
    st.session_state.last_update = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# === Sidebar: Choose Model Type ===
model_choice = st.sidebar.selectbox(
    "🔍 Choose Prediction Model:",
    ["Machine Learning (ML)", "Deep Learning (LSTM)"]
)

# === User Input ===
st.markdown('<div class="main">', unsafe_allow_html=True)
city = st.text_input("🏙️ Enter your city name:", st.session_state.city)
st.session_state.city = city    # Store for refresh reuse

col1, col2 = st.columns(2)

# === Main Predict Button ===
with col1:
    predict_clicked = st.button("🔍 Get Live Data & Predict")

# === Manual Refresh Button ===
with col2:
    refresh_clicked = st.button("🔄 Refresh Data", key="refresh_btn")

# === Prediction Logic (used by both buttons) ===
def get_live_prediction(city, model_choice):
    if not WEATHER_API_KEY:
        st.warning("⚠️ WEATHER_API_KEY not found in .env. Set it for live data.")
        st.stop()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    data = requests.get(url).json()

    if data.get("cod") == 200:
        temp = data["main"]["temp"]
        clouds = data["clouds"]["all"]
        rain = data.get("rain", {}).get("1h", 0)
        snow = data.get("snow", {}).get("1h", 0)
        now = datetime.now()
        month = now.month
        weekday = now.weekday()
        hour = now.hour

        # Cyclical time encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Prepare DataFrame
        feature_dict = {
            'temp': temp,
            'rain_1h': rain,
            'snow_1h': snow,
            'clouds_all': clouds,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'weekday': weekday,
            'month': month
        }
        features = pd.DataFrame([feature_dict])[feature_columns]

        # === Prediction using selected model ===
        if model_choice == "Machine Learning (ML)":
            traffic_pred = model.predict(features)[0]
        else:
            if dl_model is not None:
                # Scale the data (like during DL training)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform([[temp]])  # simplified scaling
                # Use dummy time steps (10) for prediction
                seq = np.array([scaled[-10:]]) if len(scaled) >= 10 else np.pad(scaled, ((10 - len(scaled), 0), (0, 0)), mode='edge')
                seq = seq.reshape((1, 10, 1))
                traffic_pred = dl_model.predict(seq)[0][0] * 7000  # Rescale approx (adjust per dataset)
            else:
                st.warning("⚠️ Deep Learning model not available.")
                traffic_pred = 0

        # Classification Prediction
        cls_pred = cls_model.predict(features)
        congestion_label = le.inverse_transform(cls_pred)[0]

        st.session_state.prediction_data = {
            "temp": temp,
            "clouds": clouds,
            "traffic_pred": int(traffic_pred),
            "congestion": congestion_label,
            "model_used": model_choice
        }
        st.session_state.last_update = now.strftime("%H:%M:%S")
    else:
        st.error("⚠️ City not found. Please try again.")

# === Trigger Prediction or Refresh ===
if predict_clicked or refresh_clicked:
    get_live_prediction(st.session_state.city, model_choice)

# === Display Prediction Results ===
if st.session_state.prediction_data:
    data = st.session_state.prediction_data
    st.metric("🌡️ Temperature (°C)", f"{data['temp']:.1f}")
    st.metric("☁️ Cloud Coverage (%)", f"{data['clouds']}%")
    st.metric("🚗 Predicted Traffic", f"{data['traffic_pred']} vehicles/hour")
    st.metric("🧠 Model Used", data["model_used"])

    color_map = {
        "Low": ("🟢 Low (Smooth Flow)", "✅ Roads are clear. Great time to travel!", "low-tip"),
        "Moderate": ("🟠 Moderate (Medium Flow)", "⚠️ Some congestion — plan ahead.", "moderate-tip"),
        "High": ("🔴 High (Heavy Congestion)", "🚨 Avoid if possible — peak hour traffic!", "high-tip")
    }

    congestion_text, travel_tip, css_class = color_map.get(data["congestion"], ("❓ Unknown", "", ""))
    st.markdown(f"### 🚦 Congestion Level: {congestion_text}")
    st.markdown(f'<div class="tip {css_class}">{travel_tip}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="refresh-info">🕐 Last Updated: {st.session_state.last_update}</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
