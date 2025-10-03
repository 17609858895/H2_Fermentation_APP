# H2_Fermentation_APP.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# ─── 页面设置 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dark Fermentation H₂ Yield Predictor",
                   layout="centered")

# ─── 自定义 CSS：浅蓝背景 + 大字号 ────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    max-width: 1100px;
    margin: auto;
    background-color: #eaf6ff;           /* 保持浅蓝背景 */
    padding: 2.2rem 2.6rem 2.8rem 2.6rem;
    border-radius: 16px;
    box-shadow: 0 0 12px rgba(0, 100, 80, 0.06);
    font-family: "Segoe UI", system-ui, -apple-system, "Noto Sans", sans-serif;
    color: #0f172a;                       /* 深色文字保证对比 */
}

#MainMenu, header, footer {visibility: hidden;}  /* 便于截图 */

.custom-header{
    font-size: 2.3rem;    /* ↑ 标题更大 */
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-bottom: .25rem;
}
.custom-sub{
    font-size: 1.15rem;
    color: #334155;
    text-align: center;
    margin-bottom: 1.6rem;
}
.section-title{
    font-size: 1.25rem;   /* ↑ 分组标题更大 */
    font-weight: 800;
    color: #0f172a;
    margin: .6rem 0 .7rem 0;
}

/* 标签字号加大、加粗 */
.stNumberInput label, .stSlider label {
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;  /* ↑ 标签字体 */
    letter-spacing: .2px;
}

/* 输入框里的数字字号更大、更粗 */
input[type="number"]{
    height: 46px !important;       /* ↑ 输入框高度 */
    font-size: 1.2rem !important;  /* ↑ 数字字号 */
    font-weight: 750 !important;   /* ↑ 数字加粗 */
    border-radius: 10px !important;
}

/* 按钮更大 */
.stButton>button, .stDownloadButton>button{
    border-radius: 10px;
    font-weight: 800;
    font-size: 1.05rem;            /* ↑ 按钮文字 */
    padding: .7rem 1.3rem;
    border: none;
}
.stButton>button{ background:#2e7d67; color:#fff; }
.stButton>button:hover{ background:#226b57; }
.stDownloadButton>button{
    background:#ffffff; color:#0f172a; border:1px solid #cbd5e1;
}
.stDownloadButton>button:hover{
    background:#f1f5f9; border-color:#94a3b8;
}

/* 高对比结果框，字号更大 */
.result-box{
    margin-top: 1.0rem;
    background: #d8f3dc;
    border-left: 8px solid #2e7d67;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    font-size: 1.35rem;            /* ↑ 结果文字 */
    font-weight: 850;
    color:#064e3b;
}

/* 说明文字 */
.small-note{ color:#334155; font-size:1rem; }
</style>
""", unsafe_allow_html=True)

# ─── 加载模型 ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("HGB_pipeline.pkl")

model = load_model()
feature_names = model.named_steps["scaler"].feature_names_in_

# ─── 标题 ─────────────────────────────────────────────────────────────────
st.markdown('<div class="custom-header">Dark Fermentation H₂ Yield Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Predict H₂ yield (mL H₂ g⁻¹ substrate) from experimental parameters</div>', unsafe_allow_html=True)

# ─── 三栏输入 ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="section-title">Catalysts & Biomass</div>', unsafe_allow_html=True)
    fe      = st.number_input("Fe (mg L⁻¹)", min_value=0.0, max_value=100.0, step=0.1, value=5.0, format="%.1f")
    ni      = st.number_input("Ni (mg L⁻¹)", min_value=0.0, max_value=50.0,  step=0.1, value=1.0, format="%.1f")
    biomass = st.number_input("Biomass (g)",  min_value=0.0, max_value=100.0, step=0.1, value=0.5, format="%.1f")

with col2:
    st.markdown('<div class="section-title">Water Chemistry</div>', unsafe_allow_html=True)
    pH  = st.number_input("pH",               min_value=0.0, max_value=14.0,  step=0.1, value=7.0,   format="%.1f")
    COD = st.number_input("COD (mg L⁻¹)",     min_value=0.0, max_value=2000.0, step=10.0, value=1000.0, format="%.0f")
    HRT = st.number_input("HRT (h)",          min_value=0.0, max_value=72.0,   step=1.0,  value=24.0,   format="%.0f")

with col3:
    st.markdown('<div class="section-title">Substrate Profile</div>', unsafe_allow_html=True)
    acetate      = st.number_input("Acetate (g L⁻¹)",            min_value=0.0, max_value=200.0, step=0.1, value=50.0, format="%.1f")
    ethanol      = st.number_input("Ethanol (g L⁻¹)",            min_value=0.0, max_value=100.0, step=0.1, value=20.0, format="%.1f")
    butyrate     = st.number_input("Butyrate (g L⁻¹)",           min_value=0.0, max_value=100.0, step=0.1, value=10.0, format="%.1f")
    ac_but_ratio = st.number_input("Acetate/Butyrate ratio (–)", min_value=0.0, max_value=10.0,  step=0.1, value=5.0,  format="%.1f")

# ─── 预测 & 下载 ─────────────────────────────────────────────────────────
prediction = None
df_result = None

btn_col, dl_col = st.columns([1.5, 1])
with btn_col:
    if st.button("Predict H₂ Yield"):
        values = [fe, ni, biomass, acetate, butyrate, ac_but_ratio, ethanol, pH, HRT, COD]
        X = pd.DataFrame([values], columns=feature_names)
        prediction = float(model.predict(X)[0])

        st.markdown(
            f'<div class="result-box">Predicted H₂ yield: {prediction:.2f} mL H₂ g⁻¹ substrate</div>',
            unsafe_allow_html=True
        )

        df_result = pd.DataFrame([{
            "Fe (mg L⁻¹)": fe,
            "Ni (mg L⁻¹)": ni,
            "Biomass (g)": biomass,
            "pH": pH,
            "COD (mg L⁻¹)": COD,
            "HRT (h)": HRT,
            "Acetate (g L⁻¹)": acetate,
            "Ethanol (g L⁻¹)": ethanol,
            "Butyrate (g L⁻¹)": butyrate,
            "Ac/But ratio (–)": ac_but_ratio,
            "Predicted H₂ (mL g⁻¹)": round(prediction, 2)
        }])

with dl_col:
    if prediction is not None and df_result is not None:
        towrite = BytesIO()
        df_result.to_csv(towrite, index=False)
        st.download_button(
            label="Download results (CSV)",
            data=towrite.getvalue(),
            file_name="H2_Fermentation_Prediction.csv",
            mime="text/csv"
        )

# ─── 页脚 ─────────────────────────────────────────────────────────────────
st.markdown("""
---
<div class="small-note">
<b>Abbreviations.</b> COD, chemical oxygen demand; HRT, hydraulic retention time.  
Model: HistGradientBoostingRegressor + StandardScaler pipeline.
</div>
""", unsafe_allow_html=True)
