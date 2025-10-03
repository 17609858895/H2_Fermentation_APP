# H2_Fermentation_APP.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# ─── 页面设置 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dark Fermentation H₂ Yield Predictor",
                   layout="centered")

# ─── 自定义 CSS（高对比、可印刷） ─────────────────────────────────────────
st.markdown("""
<style>
/* 清爽白底，黑字，高对比，适合论文截图 */
.stApp {
    max-width: 1100px;
    margin: auto;
    background: #ffffff;
    padding: 2rem 2rem 2.5rem 2rem;
    border-radius: 12px;
    box-shadow: 0 0 8px rgba(0,0,0,0.06);
    font-family: "Segoe UI", system-ui, -apple-system, "Noto Sans", sans-serif;
    color: #111;
}
/* 去掉默认顶栏/底栏，便于截图 */
#MainMenu, header, footer {visibility: hidden;}

.custom-header{
    font-size: 2.2rem; font-weight: 750; color:#111; text-align:center; margin-bottom:.25rem;
}
.custom-sub{
    font-size: 1.05rem; color:#333; text-align:center; margin-bottom:1.4rem;
}
.section-title{
    font-size: 1.15rem; font-weight:700; color:#111;
    margin: .5rem 0 .6rem 0; letter-spacing:.2px;
}

/* 表单标签与输入框：更大字号、更高可读性 */
.stNumberInput label, .stSlider label { 
    color:#111 !important; font-weight:700 !important; font-size:1rem !important;
}
input[type="number"]{
    border-radius:8px !important; height:40px !important; font-size:1.05rem !important;
}

/* 按钮 */
.stButton>button, .stDownloadButton>button{
    border-radius:10px; font-weight:700; font-size:1rem; padding:.6rem 1.2rem;
    border:1px solid #111;
}
.stButton>button{ background:#111; color:#fff; }
.stButton>button:hover{ background:#222; }

/* 预测结果：高对比白底黑字 */
.result-box{
    margin-top: 1rem;
    border: 2px solid #111; border-radius: 10px;
    padding: .9rem 1rem; text-align: center;
    font-size: 1.2rem; font-weight: 750; color:#111;
}

/* 表格/说明文字 */
.small-note{ color:#444; font-size:.92rem; }
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

# ─── 三栏输入（更统一的单位与格式） ────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="section-title">Catalysts & Biomass</div>', unsafe_allow_html=True)
    fe      = st.number_input("Fe (mg L⁻¹)", min_value=0.0, max_value=100.0, step=0.1, value=5.0, format="%.1f")
    ni      = st.number_input("Ni (mg L⁻¹)", min_value=0.0, max_value=50.0,  step=0.1, value=1.0, format="%.1f")
    biomass = st.number_input("Biomass (g)",  min_value=0.0, max_value=100.0, step=0.1, value=0.5, format="%.1f")

with col2:
    st.markdown('<div class="section-title">Water Chemistry</div>', unsafe_allow_html=True)
    # 用数字输入替代滑块，避免刻度小字
    pH  = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, value=7.0, format="%.1f")
    COD = st.number_input("COD (mg L⁻¹)", min_value=0.0, max_value=2000.0, step=10.0, value=1000.0, format="%.0f")
    HRT = st.number_input("HRT (h)",       min_value=0.0, max_value=72.0,   step=1.0,  value=24.0,   format="%.0f")

with col3:
    st.markdown('<div class="section-title">Substrate Profile</div>', unsafe_allow_html=True)
    acetate      = st.number_input("Acetate (g L⁻¹)",          min_value=0.0, max_value=200.0, step=0.1, value=50.0, format="%.1f")
    ethanol      = st.number_input("Ethanol (g L⁻¹)",          min_value=0.0, max_value=100.0, step=0.1, value=20.0, format="%.1f")
    butyrate     = st.number_input("Butyrate (g L⁻¹)",         min_value=0.0, max_value=100.0, step=0.1, value=10.0, format="%.1f")
    ac_but_ratio = st.number_input("Acetate/Butyrate ratio (–)", min_value=0.0, max_value=10.0,  step=0.1, value=5.0,  format="%.1f")

# ─── 预测 ─────────────────────────────────────────────────────────────────
prediction = None
df_result = None

left, right = st.columns([1.3, 1])
with left:
    if st.button("Predict H₂ Yield"):
        values = [fe, ni, biomass, acetate, butyrate, ac_but_ratio, ethanol, pH, HRT, COD]
        X = pd.DataFrame([values], columns=feature_names)
        prediction = float(model.predict(X)[0])

        st.markdown(
            f'<div class="result-box">Predicted H₂ yield: {prediction:.2f} mL H₂ g⁻¹ substrate</div>',
            unsafe_allow_html=True
        )

        # 结果表（单位与输入一致，便于读者核对）
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

with right:
    if prediction is not None and df_result is not None:
        buf = BytesIO()
        df_result.to_csv(buf, index=False)
        st.download_button(
            label="Download results (CSV)",
            data=buf.getvalue(),
            file_name="H2_Fermentation_Prediction.csv",
            mime="text/csv"
        )

# ─── 图注式说明（缩写释义，便于论文读者） ───────────────────────────────────
st.markdown("""
---
<div class="small-note">
<b>Abbreviations.</b> COD, chemical oxygen demand; HRT, hydraulic retention time.  
Model: HistGradientBoostingRegressor + StandardScaler pipeline.
</div>
""", unsafe_allow_html=True)
