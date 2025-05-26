# H2_Fermentation_APP.py

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ─── 页面配置 ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Fermentation H₂ Yield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 自定义样式（仅修改背景为浅蓝色） ─────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp {
        max-width: 1100px;
        margin: auto;
        background-color: #eaf6ff;  /* 淡蓝色背景 */
        padding: 2.5rem 3rem 3.5rem 3rem;
        border-radius: 18px;
        box-shadow: 0px 0px 12px rgba(0, 100, 80, 0.06);
    }
    html, body, [class*="css"] {
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

# ─── 页面标题与副标题 ───────────────────────────────────────────────────────────
st.markdown('<div class="custom-header">💧 Dark Fermentation H₂ Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Predict hydrogen yield (mL H₂/g substrate) from your input parameters</div>', unsafe_allow_html=True)

# ─── 侧边栏输入参数 ──────────────────────────────────────────────────────────────
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

# ─── 加载模型 ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("HGB_pipeline.pkl")

model = load_model()

# ─── 预测与结果展示 ───────────────────────────────────────────────────────────────
if predict_button:
    X = pd.DataFrame([{
        "fe": fe,
        "ni": ni,
        "biomass": biomass,
        "pH": pH,
        "cod": COD,
        "hrt": HRT,
        "acetate": acetate,
        "ethanol": ethanol,
        "butyrate": butyrate,
        "acetate_butyrate": ac_but_ratio
    }])

    y_pred = model.predict(X)[0]
    st.metric(
        label="🔬 Predicted H₂ Yield (mL H₂/g)",
        value=f"{y_pred:.2f}",
        delta=None
    )

# ─── 页脚 ─────────────────────────────────────────────────────────────────────────
st.markdown("""
---
<small style="color:#666;">
Data: 210 literature‐sourced points (see GitHub).  
Model: HistGradientBoostingRegressor + StandardScaler.
</small>
""", unsafe_allow_html=True)
