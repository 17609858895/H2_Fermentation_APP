# H2_Fermentation_APP.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# ─── 页面设置 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Fermentation H₂ Yield Predictor",
    layout="centered"
)

# ─── 自定义 CSS 样式 ───────────────────────────────────────────────────────────
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
        color: #1b4332;
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
        color: #2d6a4f;
    }
    input[type="number"] {
        border-radius: 6px !important;
        height: 38px !important;
        font-size: 0.95rem !important;
    }
    .stButton>button, .stDownloadButton>button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.55rem 1.2rem;
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
    .stSuccess {
        background-color: #d8f3dc;
        color: #065f46;
        padding: 1rem;
        border-left: 6px solid #40916c;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.15rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ─── 加载模型 ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("HGB_pipeline.pkl")

model = load_model()

# 获取训练时的列名顺序（确保匹配）
feature_names = model.named_steps["scaler"].feature_names_in_

# ─── 页面标题 ─────────────────────────────────────────────────────────────────
st.markdown('<div class="custom-header">💧 Dark Fermentation H₂ Yield Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Predict H₂ yield (mL H₂/g substrate) from experimental parameters</div>', unsafe_allow_html=True)

# ─── 三栏输入 ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="section-title">🧪 Catalysts & Biomass</div>', unsafe_allow_html=True)
    fe      = st.number_input("Fe concentration (mg/L)",      0.0, 100.0, step=0.1, value=5.0)
    ni      = st.number_input("Ni concentration (mg/L)",      0.0, 50.0,  step=0.1, value=1.0)
    biomass = st.number_input("Biomass (g)",       0.0, 100.0,  step=1.0, value=0.5)

with col2:
    st.markdown('<div class="section-title">💧 Water Chemistry</div>', unsafe_allow_html=True)
    pH  = st.slider("pH", 0.0, 14.0, 7.0, step=0.1)
    COD = st.number_input("COD (mg/L)",    0.0, 2000.0, step=10.0, value=1000.0)
    HRT = st.number_input("HRT (h)",       0.0, 72.0,   step=1.0,  value=24.0)

with col3:
    st.markdown('<div class="section-title">⚗️ Substrate Profile</div>', unsafe_allow_html=True)
    acetate      = st.number_input("Acetate (g/L)",           0.0, 200.0, step=1.0, value=50.0)
    ethanol      = st.number_input("Ethanol (g/L)",           0.0, 100.0, step=1.0, value=20.0)
    butyrate     = st.number_input("Butyrate (g/L)",          0.0, 100.0, step=1.0, value=10.0)
    ac_but_ratio = st.number_input("Acetate/Butyrate ratio", 0.0, 10.0,  step=0.1, value=5.0)

# ─── 预测与下载 ───────────────────────────────────────────────────────────────
prediction = None
df_result = None

btn_col, dl_col = st.columns([1.5, 1])
with btn_col:
    if st.button("🔍 Predict H₂ Yield"):
        values = [fe, ni, biomass, acetate, butyrate, ac_but_ratio, ethanol, pH, HRT, COD]
        X = pd.DataFrame([values], columns=feature_names)
        prediction = model.predict(X)[0]
        st.success(f"✅ Predicted H₂ Yield: **{prediction:.2f} mL H₂/g**")

        df_result = pd.DataFrame([{
            "Fe (mg/L)": fe, "Ni (mg/L)": ni, "Biomass (g VS/g)": biomass,
            "pH": pH, "COD (mg/L)": COD, "HRT (h)": HRT,
            "Acetate (mM)": acetate, "Ethanol (mM)": ethanol,
            "Butyrate (mM)": butyrate, "Ac/But Ratio": ac_but_ratio,
            "Predicted H₂ (mL/g)": round(prediction, 2)
        }])

with dl_col:
    if prediction is not None and df_result is not None:
        towrite = BytesIO()
        df_result.to_csv(towrite, index=False)
        st.download_button(
            label="📁 Download Results as CSV",
            data=towrite.getvalue(),
            file_name="H2_Fermentation_Prediction.csv",
            mime="text/csv"
        )

# ─── 页脚 ─────────────────────────────────────────────────────────────────────
st.markdown("""
---
<small style="color:#666;"> 
Model: HistGradientBoostingRegressor + StandardScaler pipeline.
</small>
""", unsafe_allow_html=True)
