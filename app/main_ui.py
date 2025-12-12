import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add the project root to system path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ARTIFACTS_DIR

# --- CONFIGURATION ---
# We use the Random Forest model as it had the highest accuracy (90.38%)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, 'feature_names.pkl')

# --- PAGE SETUP ---
st.set_page_config(page_title="ImpactSense", page_icon="🌍", layout="centered")

# Custom CSS for Professional Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
    }
    .stButton>button:hover { background-color: #0b5ed7; }
    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        color: white;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe { background-color: #198754; } /* Green */
    .caution { background-color: #ffc107; color: #000 !important; } /* Yellow */
    .danger { background-color: #fd7e14; } /* Orange */
    .critical { background-color: #dc3545; } /* Red */
    </style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error(f"❌ Critical Error: Artifacts not found at {ARTIFACTS_DIR}. Please run 'python main.py' first.")
        st.stop()

model, scaler, feature_names = load_resources()

# --- HEADER ---
st.title("🌍 ImpactSense")
st.markdown("#### AI-Powered Earthquake Risk Assessment")
st.markdown("---")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Earthquake_wave_animation.gif/220px-Earthquake_wave_animation.gif", width='stretch')
    st.info("**Model Accuracy:** 90.38% (Random Forest)")
    st.markdown("### How it works")
    st.caption("This system uses seismic physics and machine learning to predict the damage potential of an earthquake based on magnitude, depth, and intensity reports.")

# --- INPUT FORM ---
st.subheader("📝 Enter Seismic Parameters")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        magnitude = st.number_input("Magnitude (Richter)", 0.0, 10.0, 6.5, 0.1, help="The seismic energy released.")
        depth = st.number_input("Depth (km)", 0.0, 1000.0, 30.0, 1.0, help="Depth of the hypocenter.")
        cdi = st.number_input("CDI (Reported Intensity)", 0.0, 12.0, 5.0, help="Community Decimal Intensity.")
        
    with col2:
        mmi = st.number_input("MMI (Instrument Intensity)", 0.0, 12.0, 6.0, help="Modified Mercalli Intensity.")
        sig = st.number_input("Significance Score", 0.0, 1000.0, 100.0, help="Net impact score.")
    
    submit = st.form_submit_button("🚀 Analyze Risk")

# --- PREDICTION LOGIC ---
if submit:
    # 1. Feature Engineering (Replicating src.preprocessing logic)
    energy_release = 10 ** (1.5 * magnitude)
    impact_factor = magnitude / np.log1p(depth) if depth > 0 else magnitude
    is_shallow = 1 if depth < 70 else 0
    
    # 2. DataFrame Creation
    input_data = pd.DataFrame({
        'magnitude': [magnitude],
        'depth': [depth],
        'cdi': [cdi],
        'mmi': [mmi],
        'sig': [sig],
        'energy_release': [energy_release],
        'impact_factor': [impact_factor],
        'is_shallow': [is_shallow]
    })
    
    # 3. Reorder Columns
    input_data = input_data[feature_names]
    
    # 4. Scale & Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    
    # 5. Output
    labels = ["Green Alert (Low Risk)", "Yellow Alert (Caution)", "Orange Alert (High Risk)", "Red Alert (Critical)"]
    css_classes = ["safe", "caution", "danger", "critical"]
    
    result_label = labels[prediction]
    result_class = css_classes[prediction]
    confidence = np.max(probs) * 100
    
    st.markdown(f'<div class="result-box {result_class}">{result_label}</div>', unsafe_allow_html=True)
    st.write("")
    
    # Metrics breakdown
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Confidence", f"{confidence:.1f}%")
    kpi2.metric("Impact Factor", f"{impact_factor:.2f}")
    kpi3.metric("Energy (J)", f"{energy_release:.1e}")