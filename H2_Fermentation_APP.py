# H2_Fermentation_APP.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# ─── 页面设置 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Fermentation H₂ Yield Predictor",
    layout="centered"
)

# ─── 自定义 CSS：加宽、不换行、放大字体、保留浅蓝背景与图标 ───────────────
st.markdown("""
<style>
.stApp {
    max-width: 1360px;               /* ↑ 稍微再加宽，避免长标签换行 */
    margin: auto;
    background-color: #eaf6ff;       /* 浅蓝背景保留 */
    padding: 2.4rem 3rem 3.2rem 3rem;
    border-radius: 16px;
    box-shadow: 0 0 12px rgba(0, 100, 80, 0.06);
    font-family: "Segoe UI", system-ui, -apple-system, "Noto Sans", sans-serif;
    color: #0f172a;
}
#MainMenu, header, footer {visibility: hidden;}  /* 便于截图 */

.custom-header{
    font-size: 2.35rem;              /* 顶部主标题 */
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-bottom: .25rem;
}
.custom-sub{
    font-size: 1.18rem;              /* 副标题稍大 */
    color: #334155;
    text-align: center;
    margin-bottom: 1.6rem;
}

/* 分组标题（含🧪 💧 ⚗️），不换行 */
.section-title{
    font-size: 1.38rem;              /* ↑ 分组标题更大 */
    font-weight: 800;
    color: #0f172a;
    margin: .6rem 0 .7rem 0;
    white-space: nowrap;             /* 不换行，如“Catalysts & Biomass” */
}

/* 放大每个输入特征标签（如 Fe (mg L⁻¹)），并避免换行 */
.stNumberInput label,
.stNumberInput > label > div,
.stSlider label,
.stSlider > label > div{
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 1.38rem !important;   /* ← 标签显著变大 */
    letter-spacing: .2px;
    white-space: nowrap;             /* 不换行，如 Acetate/Butyrate ratio (–) */
}

/* 输入框中的数字：更大，但不加粗（保持清晰不臃肿） */
input[type="number"]{
    height: 50px !important;
    font-size: 1.32rem !important;   /* ← 数字再加大 */
    font-weight: 400 !important;     /* 不加粗 */
    border-radius: 10px !important;
}

/* 按钮与下载按钮 */
.stButton>button, .stDownloadButton>button{
    border-radius: 10px;
    font-weight: 800;
    font-size: 1.08rem;
    padding: .72rem 1.35rem;
    border: none;
}
.stButton>button{ background:#2e7d67; color:#fff; }
.stButton>button:hover{ background:#226b57; }
.stDownloadButton>button{ background:#ffffff; color:#0f172a; border:1px solid #cbd5e1; }
.stDownloadButton>button:hover{ background:#f1f5f9; border-color:#94a3b8; }

/* 预测结果框：高对比、字号更大 */
.result-box{
    margin-top: 1.0rem;
    background: #d8f3dc;
    border-left: 8px solid #2e7d67;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    font-size: 1.40rem;
    font-weight: 850;
    color:#064e3b;
}

/* 页脚小字也放大 */
.small-note{
    color:#334155;
    font-size: 1.14rem;              /* ← 说明文字更大 */
    line-height: 1.58;
}
</style>
""", unsafe_allow_html=True)

# ─── 加载模型 ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("HGB_pipeline.pkl")

model = load_model()

# 获取训练时的列名顺序（确保匹配）
feature_names = model.named_steps["scaler"].feature_names_in_

# ─── 页面标题（保留 💧 图标） ────────────────────────────────────────────
st.markdown('<div class="custom-header">💧 Dark Fermentation H₂ Yield Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-sub">Predict H₂ yield (mL H₂ g⁻¹ substrate) from experimental parameters</div>', unsafe_allow_html=True)

# ─── 三栏输入（保留 🧪 💧 ⚗️ 图标） ──────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="section-title">🧪 Catalysts & Biomass</div>', unsafe_allow_html=True)
    fe      = st.number_input("Fe (mg L⁻¹)", min_value=0.0, max_value=100.0, step=0.1, value=5.0, format="%.1f")
    ni      = st.number_input("Ni (mg L⁻¹)", min_value=0.0, max_value=50.0,  step=0.1, value=1.0, format="%.1f")
    biomass = st.number_input("Biomass (g)",  min_value=0.0, max_value=100.0, step=0.1, value=0.5, format="%.1f")

with col2:
    st.markdown('<div class="section-title">💧 Water Chemistry</div>', unsafe_allow_html=True)
    # 如需滑块可改：st.slider("pH", 0.0, 14.0, 7.0, step=0.1)
    pH  = st.number_input("pH",               min_value=0.0, max_value=14.0,  step=0.1, value=7.0,   format="%.1f")
    COD = st.number_input("COD (mg L⁻¹)",     min_value=0.0, max_value=2000.0, step=10.0, value=1000.0, format="%.0f")
    HRT = st.number_input("HRT (h)",          min_value=0.0, max_value=72.0,   step=1.0,  value=24.0,   format="%.0f")

with col3:
    st.markdown('<div class="section-title">⚗️ Substrate Profile</div>', unsafe_allow_html=True)
    acetate      = st.number_input("Acetate (g L⁻¹)",            min_value=0.0, max_value=200.0, step=0.1, value=50.0, format="%.1f")
    ethanol      = st.number_input("Ethanol (g L⁻¹)",            min_value=0.0, max_value=100.0, step=0.1, value=20.0, format="%.1f")
    butyrate     = st.number_input("Butyrate (g L⁻¹)",           min_value=0.0, max_value=100.0, step=0.1, value=10.0, format="%.1f")
    ac_but_ratio = st.number_input("Acetate/Butyrate ratio (–)", min_value=0.0, max_value=10.0,  step=0.1, value=5.0,  format="%.1f")

# ─── 预测 & 下载 ─────────────────────────────────────────────────────────
prediction = None
df_result = None

btn_col, dl_col = st.columns([1.5, 1])
with btn_col:
    if st.button("🔍 Predict H₂ Yield"):
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
        buff = BytesIO()
        df_result.to_csv(buff, index=False)
        st.download_button(
            label="📁 Download Results as CSV",
            data=buff.getvalue(),            # ← 修正后的变量名
            file_name="H2_Fermentation_Prediction.csv",
            mime="text/csv"
        )

# ─── 页脚（说明文字加大） ─────────────────────────────────────────────────
st.markdown("""
---
<div class="small-note">
<b>Abbreviations.</b> COD, chemical oxygen demand; HRT, hydraulic retention time.  
Model: HistGradientBoostingRegressor + StandardScaler pipeline.
</div>
""", unsafe_allow_html=True)
