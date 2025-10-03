# H2_Fermentation_APP.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# --- Page Settings ---
st.set_page_config(
    page_title="Dark Fermentation H₂ Yield Predictor",
    layout="centered"
)

# --- Custom CSS Styling (Further Optimized for Readability) ---
st.markdown("""
    <style>
    /* --- General App Styling --- */
    .stApp {
        max-width: 1100px;
        margin: auto;
        background-color: #f0f8ff;
        padding: 2.5rem 3rem 3.5rem 3rem;
        border-radius: 18px;
        box-shadow: 0px 4px 15px rgba(0, 80, 120, 0.08);
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }

    /* --- Headers --- */
    .custom-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0d3b66;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .custom-sub {
        font-size: 1.15rem;
        color: #333;
        text-align: center;
        margin-bottom: 2.5rem;
    }

    /* --- Section Titles --- */
    .section-title {
        font-size: 1.35rem; /* 微调字号以防止换行 */
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1.2rem;
        color: #1e6091;
        border-bottom: 2px solid #ddebf8;
        padding-bottom: 0.4rem;
        white-space: nowrap; /* 强制标题不换行 */
    }

    /* --- Parameter Label Styling (FINAL ADJUSTMENT) --- */
    .stNumberInput > label > div,
    .stSlider > label > div {
        color: #0d3b66 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important; /* 再次增大标签字体 */
    }

    /* --- Input/Button Elements --- */
    input[type="number"] {
        border-radius: 6px !important;
        height: 40px !important;
        font-size: 1rem !important;
        border: 1px solid #ccc;
    }
    .stButton>button, .stDownloadButton>button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.6rem 1.3rem;
        width: 100%;
    }
    .stButton>button {
        background-color: #52b788;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #40916c;
    }
    .stDownloadButton>button {
        background-color: #ffffff;
        color: #333;
        border: 1px solid #ccc;
    }
    .stDownloadButton>button:hover {
        background-color: #eef7f2;
        border-color: #88cbb3;
    }

    /* --- Success Message --- */
    .stSuccess {
        background-color: #e6f9f1;
        color: #065f46;
        padding: 1.1rem;
        border-left: 6px solid #40916c;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    
    /* --- Footer (FINAL ADJUSTMENT) --- */
    .footer-text {
        color: #333 !important; /* 深色页脚文本 */
        font-size: 1.0rem !important; /* 显著增大页脚字体 */
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_model():
    # Make sure the model file "HGB_pipeline.pkl" is in the same directory
    return joblib.load("HGB_pipeline.pkl")

model = load_model()

# Ensure feature names are correctly ordered as during training
try:
    feature_names = model.named_steps["scaler"].get_feature_names_out()
except AttributeError:
    # Fallback for older scikit-learn versions
    feature_names = model.named_steps["scaler"].feature_names_in_


# --- Page Title ---
st.markdown('<div class="custom-header">💧 Dark Fermentation H₂ Yield Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Predict H₂ yield (mL H₂/g substrate) from experimental parameters</div>', unsafe_allow_html=True)


# --- Input Columns ---
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="section-title">🧪 Catalysts & Biomass</div>', unsafe_allow_html=True)
    fe      = st.number_input("Fe concentration (mg/L)", 0.0, 100.0, step=0.1, value=5.0)
    ni      = st.number_input("Ni concentration (mg/L)", 0.0, 50.0,  step=0.1, value=1.0)
    biomass = st.number_input("Biomass (g)",             0.0, 100.0, step=1.0, value=0.5)

with col2:
    st.markdown('<div class="section-title">💧 Water Chemistry</div>', unsafe_allow_html=True)
    pH  = st.slider("pH", 0.0, 14.0, 7.0, step=0.1)
    COD = st.number_input("COD (mg/L)", 0.0, 2000.0, step=10.0, value=1000.0)
    HRT = st.number_input("HRT (h)",    0.0, 72.0,   step=1.0,  value=24.0)

with col3:
    st.markdown('<div class="section-title">⚗️ Substrate Profile</div>', unsafe_allow_html=True)
    acetate      = st.number_input("Acetate (g/L)",          0.0, 200.0, step=1.0, value=50.0)
    ethanol      = st.number_input("Ethanol (g/L)",          0.0, 100.0, step=1.0, value=20.0)
    butyrate     = st.number_input("Butyrate (g/L)",         0.0, 100.0, step=1.0, value=10.0)
    ac_but_ratio = st.number_input("Acetate/Butyrate ratio", 0.0, 10.0,  step=0.1, value=5.0)


# --- Prediction and Download Logic ---
# Initialize session state for prediction and dataframe
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'df_result' not in st.session_state:
    st.session_state.df_result = None

# Use columns for button layout
left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    if st.button("🔍 Predict H₂ Yield"):
        # Gather values
        values_dict = {
            'Fe concentration (mg/L)': fe,
            'Ni concentration (mg/L)': ni,
            'Biomass (g)': biomass,
            'Acetate (g/L)': acetate,
            'Butyrate (g/L)': butyrate,
            'Acetate/Butyrate ratio': ac_but_ratio,
            'Ethanol (g/L)': ethanol,
            'pH': pH,
            'HRT (h)': HRT,
            'COD (mg/L)': COD
        }
        
        # Create DataFrame with columns in the correct order
        input_data = pd.DataFrame([values_dict])[feature_names]
        
        # Predict and store in session state
        prediction_value = model.predict(input_data)[0]
        st.session_state.prediction = prediction_value
        
        st.session_state.df_result = pd.DataFrame([{
            "Fe (mg/L)": fe,
            "Ni (mg/L)": ni,
            "Biomass (g)": biomass,
            "pH": pH,
            "COD (mg/L)": COD,
            "HRT (h)": HRT,
            "Acetate (g/L)": acetate,
            "Ethanol (g/L)": ethanol,
            "Butyrate (g/L)": butyrate,
            "Ac/But Ratio": ac_but_ratio,
            "Predicted H₂ Yield (mL H₂/g)": round(prediction_value, 2)
        }])

# Display prediction result if it exists
if st.session_state.prediction is not None:
    st.success(f"✅ Predicted H₂ Yield: **{st.session_state.prediction:.2f} mL H₂/g**")

# Display download button if result exists
if st.session_state.df_result is not None:
    towrite = BytesIO()
    st.session_state.df_result.to_csv(towrite, index=False, encoding='utf-8')
    towrite.seek(0)
    
    with center_col:
        st.download_button(
            label="📁 Download Results as CSV",
            data=towrite,
            file_name="H2_Fermentation_Prediction.csv",
            mime="text/csv"
        )

# --- Footer ---
st.markdown("""
---
<div class="footer-text">
Model: HistGradientBoostingRegressor + StandardScaler pipeline.
</div>
""", unsafe_allow_html=True)
