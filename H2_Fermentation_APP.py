# H2_Fermentation_APP.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO

# ─── 页面设置 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dark Fermentation H₂ Yield Predictor",
                   layout="centered")

# ─── 自定义 CSS：加宽、不换行、放大字体 ──────────────────────────────────
st.markdown("""
<style>
.stApp {
    max-width: 1280px;               /* ↑ 稍微加宽，避免换行 */
    margin: auto;
    background-color: #eaf6ff;       /* 浅蓝背景保留 */
    padding: 2.2rem 2.8rem 3rem 2.8rem;
    border-radius: 16px;
    box-shadow: 0 0 12px rgba(0, 100, 80, 0.06);
    font-family: "Segoe UI", system-ui, -apple-system, "Noto Sans", sans-serif;
    color: #0f172a;
}
#MainMenu, header, footer {visibility: hidden;}

.custom-header{
    font-size: 2.3rem;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-bottom: .25rem;
}
.custom-sub{
    font-size: 1.18rem;
    color: #334155;
    text-align: center;
    margin-bottom: 1.6rem;
}
.section-title{
    font-size: 1.35rem;              /* ↑ 分组标题更大 */
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
    font-size: 1.35rem !important;   /* ← 标签显著变大 */
    letter-spacing: .2px;
    white-space: nowrap;             /* 不换行，如 Acetate/Butyrate ratio (–) */
}

/* 输入框中的数字：更大，但不加粗 */
input[type="number"]{
    height: 48px !important;
    font-size: 1.30rem !important;   /* ← 数字再加大 */
    font-weight: 400 !important;     /* 不加粗 */
    border-radius: 10px !important;
}

/* 按钮与下载 */
.stButton>button, .stDownloadButton>button{
    border-radius: 10px;
    font-weight: 800;
    font-size: 1.08rem;
    padding: .7rem 1.3rem;
    border: none;
}
.stButton>button{ background:#2e7d67; color:#fff; }
.stButton>button:hover{ background:#226b57; }
.stDownloadButton>button{ background:#ffffff; color:#0f172a; border:1px solid #cbd5e1; }
.stDownloadButton>button:hover{ background:#f1f5f9; border-col
