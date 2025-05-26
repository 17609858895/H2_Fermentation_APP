# H2_Fermentation_APP.py

import streamlit as st
import numpy as np
import joblib

# ─── Custom CSS for styling ──────────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Header banner */
    .stApp > header {background-color: #0E1117;}
    /* Center title and subtitle */
    .app-header {text-align: center; padding: 1rem;}
    .app-header h1 {color: #FF4B4B; font-size: 3rem; margin-bottom: 0;}
    .app-header p {color: #CCCCCC; margin-top: 0.25rem;}
    /* Styled metric */
    .stMetric > div {padding: 1rem; border-radius: 0.75rem; background: rgba(255,75,75,0.1);}
    /* Sidebar header */
    .sidebar .sidebar-content {background-color: #FAFAFA;}
    </style>
""", unsafe_allow_html=True)

# ─── Page config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Fermentation H₂ Yield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Header ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header">'
            '<h1>💧 Dark Fermentation H₂ Predictor</h1>'
            '<p>Predict hydrogen yield (mL H₂/g substrate) from your input parameters</p>'
            '</div>',
            unsafe_allow_html=True)

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
    X = np.array([[fe, ni, biomass, pH, COD, HRT, acetate, ethanol, butyrate, ac_but_ratio]])
    y_pred = model.predict(X)[0]
    # Big metric in main area
    st.metric(
        label="🔬 Predicted H₂ Yield (mL H₂/g)",
        value=f"{y_pred:.2f}",
        delta=None
    )

# ─── Footer ────────────────────────────────────────────────────────────────────────
st.markdown("""
---
<small style="color:#666;">
Data: 210 literature‐sourced points (see GitHub).  
Model: HistGradientBoostingRegressor + StandardScaler.
</small>
""", unsafe_allow_html=True)
