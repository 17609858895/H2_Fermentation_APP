# H2_Fermentation_APP.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ─── Page config: must be at the top ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Fermentation H₂ Yield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp {
        max-width: 1100px;
        margin: auto;
        background-color: #eaf6ff;
        padding: 2.5rem 3rem 3.5rem 3rem;
        border-radius: 18px;
        box-shadow: 0px 0px 12px rgba(0, 100, 80, 0.06);
        font-family: 'Segoe UI', sans-serif;
    }
    .custom-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1b4965;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .custom-sub {
        font-size: 1.1rem;
        color: #4b4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #1d3557;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #277da1;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.55rem 1.2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b4965;
    }
    .stMetric > div {
        padding: 1rem;
        border-radius: 0.75rem;
        background: rgba(39,125,161,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="custom-header">💧 Dark Fermentation H₂ Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Predict hydrogen yield (mL H₂/g substrate) based on experimental inputs</div>', unsafe_allow_html=True)

# ─── Sidebar inputs ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Experiment Parameters")
    with st.expander("Metal Catalysts & Biomass"):
        fe      = st.number_input("Fe (mg/L)",      0.0, 100.0, step=0.1, value=5.0)
        ni      = st.number_input("Ni (mg/L)",      0.0, 50.0,  step=0.1, value=1.0)
        biomass = st.number_input("Biomass ratio (g VS/g)", 0.0, 1.0, step=0.01, value=0.5)
    with st.expander("Water Chemistry"):
        pH       = st.slider("pH", 0.0, 14.0, 7.0, step=0.1)
        COD      = st.number_input("COD (mg/L)",    0.0, 2000.0, step=10.0, value=1000.0)
    with st.expander("Substrate Profile"):
        acetate      = st.number_input("Acetate (mM)",      0.0, 200.0, step=1.0, value=50.0)
        ethanol      = st.number_input("Ethanol (mM)",      0.0, 100.0, step=1.0, value=20.0)
        butyrate     = st.number_input("Butyrate (mM)",     0.0, 100.0, step=1.0, value=10.0)
        ac_but_ratio = st.number_input("Acetate/Butyrate ratio", 0.0, 10.0, step=0.1, value=5.0)
    with st.expander("Process Conditions"):
        HRT = st.number_input("HRT (hours)", 0.0, 72.0, step=1.0, value=24.0)
    st.markdown("---")
    predict_button = st.button("🚀 Predict H₂ Yield")

# ─── Load model ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("HGB_pipeline.pkl")

model = load_model()

# ─── Prediction & Display ─────────────────────────────────────────────────────────
if predict_button:
    # ⬅ 用 DataFrame 带列名，避免 StandardScaler 特征错配
    X = pd.DataFrame([{
        "Fe": fe,
        "Ni": ni,
        "Biomass": biomass,
        "pH": pH,
        "COD": COD,
        "HRT": HRT,
        "Acetate": acetate,
        "Ethanol": ethanol,
        "Butyrate": butyrate,
        "Ac/But": ac_but_ratio
    }])
    y_pred = model.predict(X)[0]

    st.metric(
        label="🔬 Predicted H₂ Yield (mL H₂/g)",
        value=f"{y_pred:.2f}",
        delta=None
    )

# ─── Footer ────────────────────────────────────────────────────────────────────────
st.markdown("""
---
<small style="color:#666;">
Model: HistGradientBoostingRegressor + StandardScaler  
Data: 210 literature-sourced entries  
Developed with ❤️ using Streamlit
</small>
""", unsafe_allow_html=True)
