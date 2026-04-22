import os
import streamlit as st
import requests
import base64
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Traffic Violation Detector",
    page_icon="🚦",
    layout="centered", # Centered is MUCH more stable in iframes
)

# ─────────────────────────────────────────────────────────────────────────────
# Simplified Stable CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .stApp { background-color: #0b0f19 !important; color: #e2e8f0 !important; }
    .stat-card {
        background: #1a2236;
        border: 1px solid #243050;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sc-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; margin-bottom: 5px; }
    .sc-value { font-size: 2rem; font-weight: 800; color: #fff; }
    .hist-row {
        background: #1a2236;
        border: 1px solid #243050;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: .5rem;
        display: flex; align-items: center; justify-content: space-between;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# App Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚦 Traffic Violation Detector")
st.caption("AI-powered helmet & triple-riding detection · Powered by YOLOv8")

# ─────────────────────────────────────────────────────────────────────────────
# Main Controls (No Sidebar)
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    c1, c2 = st.columns([2, 1])
    with c1:
        file_type = st.radio("Select Type", ["Image", "Video"], horizontal=True)
        uploaded_file = st.file_uploader(f"Upload {file_type}", type=["jpg", "jpeg", "png", "mp4"])
    with c2:
        conf_threshold = st.slider("Confidence", 0.1, 0.9, 0.4)
        run_btn = st.button("🚀 Run Analysis", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Logic & Results
# ═══════════════════════════════════════════════════════════════════════════════
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if uploaded_file:
    if file_type == "Image":
        if run_btn:
            with st.spinner("Analyzing..."):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    r = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        img_bytes = base64.b64decode(data["image_base64"])
                        img_arr = np.frombuffer(img_bytes, np.uint8)
                        result_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        
                        st.session_state.last_result = {
                            "image": result_img,
                            "data": data.get("motorcycles", [])
                        }
                    else:
                        st.error("Backend Error")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Persistent Display
        if st.session_state.last_result:
            st.divider()
            st.subheader("✅ Detection Result")
            st.image(st.session_state.last_result["image"], use_container_width=True)
            
            mc_data = st.session_state.last_result["data"]
            st.subheader("📊 Summary")
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f'<div class="stat-card"><div class="sc-label">Bikes</div><div class="sc-value">{len(mc_data)}</div></div>', unsafe_allow_html=True)
            with sc2:
                viols = sum(1 for m in mc_data if m["violation"] != "Safe")
                st.markdown(f'<div class="stat-card"><div class="sc-label">Violations</div><div class="sc-value" style="color:#ef4444">{viols}</div></div>', unsafe_allow_html=True)
            with sc3:
                safe = sum(sum(1 for h in m["helmets"] if h) for m in mc_data)
                st.markdown(f'<div class="stat-card"><div class="sc-label">Safe Riders</div><div class="sc-value" style="color:#22c55e">{safe}</div></div>', unsafe_allow_html=True)

            if len(mc_data) > 0:
                with st.expander("📋 View Detailed Logs"):
                    df = pd.DataFrame(mc_data)
                    st.dataframe(df, use_container_width=True)

    elif file_type == "Video":
        if run_btn:
            with st.spinner("Processing video..."):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
                try:
                    r = requests.post(f"{API_URL}/predict", files=files, timeout=300)
                    if r.status_code == 200:
                        with open("out.mp4", "wb") as f: f.write(r.content)
                        st.success("Done!")
                        st.video("out.mp4")
                    else:
                        st.error("Video Error")
                except Exception as e:
                    st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# History Section
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Detection History")
try:
    hr = requests.get(f"{API_URL}/history", timeout=5)
    history = hr.json().get("history", []) if hr.status_code == 200 else []
    for record in history:
        st.markdown(
            f"""
            <div class="hist-row">
                <span>{record['filename']} - {record['timestamp']}</span>
                <span style="color:{'#f87171' if record['violations'] > 0 else '#4ade80'}">
                    {record['violations']} Violations
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )
except:
    st.info("Start the backend to see history.")
