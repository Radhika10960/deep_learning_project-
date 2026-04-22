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
# Config  –  set BACKEND_URL env var in production (e.g. Render/Railway URL)
# ─────────────────────────────────────────────────────────────────────────────
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ViolaTraffic · Smart Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS – premium dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ──────────────────────────────────────── */
:root {
    --bg:        #0b0f19;
    --surface:   #131929;
    --surface2:  #1a2236;
    --border:    #243050;
    --accent:    #3b82f6;
    --accent2:   #6366f1;
    --danger:    #ef4444;
    --success:   #22c55e;
    --warn:      #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    14px;
}

/* ── App-wide resets ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* ── Logo / brand bar inside sidebar ────────────────────── */
.brand-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2236 100%);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1.4rem;
    border: 1px solid var(--border);
    text-align: center;
}
.brand-box .brand-icon { font-size: 2rem; }
.brand-box .brand-title {
    font-size: 1.15rem; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.brand-box .brand-sub { font-size: 0.72rem; color: var(--muted); margin: 0; }

/* ── Page header ─────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2040 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    margin-bottom: 1.8rem;
    display: flex; align-items: center; gap: 1rem;
}
.page-header .ph-icon { font-size: 2.4rem; }
.page-header .ph-title {
    font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa 0%, #818cf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.page-header .ph-sub { color: var(--muted); font-size: 0.88rem; margin: 0; }

/* ── Stat cards ──────────────────────────────────────────── */
.stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.stat-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #6366f1); }
.stat-card.red::before   { background: linear-gradient(90deg, #ef4444, #f97316); }
.stat-card.green::before { background: linear-gradient(90deg, #22c55e, #10b981); }
.stat-card.warn::before  { background: linear-gradient(90deg, #f59e0b, #eab308); }

.stat-card .sc-label { font-size: 0.75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: .3rem; }
.stat-card .sc-value { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.stat-card .sc-icon   { font-size: 1.2rem; }

/* ── Section headings ────────────────────────────────────── */
.section-heading {
    font-size: 1rem; font-weight: 700; color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 0.7rem; margin: 1.5rem 0 0.9rem;
}

/* ── Upload zone ─────────────────────────────────────────── */
.upload-hint {
    background: var(--surface2);
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 1.4rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* ── Violation badge ─────────────────────────────────────── */
.vbadge {
    display: inline-block;
    padding: .25rem .75rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
}
.vbadge.safe    { background: rgba(34,197,94,.15);  color: #4ade80; border: 1px solid rgba(34,197,94,.3);  }
.vbadge.danger  { background: rgba(239,68,68,.15);  color: #f87171; border: 1px solid rgba(239,68,68,.3);  }

/* ── History table ───────────────────────────────────────── */
.hist-row {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: .7rem;
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem;
    transition: background .15s;
}
.hist-row:hover { background: #1e2d48; }
.hist-row .hr-name  { font-weight: 600; font-size: .9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }
.hist-row .hr-ts    { font-size: .75rem; color: var(--muted); }
.hist-row .hr-chip  { font-size: .75rem; font-weight: 600; padding: .2rem .6rem; border-radius: 6px; }
.hist-row .hr-chip.img  { background: rgba(99,102,241,.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,.3); }
.hist-row .hr-chip.vid  { background: rgba(245,158,11,.15); color: #fcd34d; border: 1px solid rgba(245,158,11,.3); }

/* ── Dataframe tweaks ────────────────────────────────────── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }

/* ── Button overrides ────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: .5rem 1.2rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Radio & selectbox ───────────────────────────────────── */
div[data-testid="stRadio"] label,
div[data-testid="stSelectbox"] label { color: var(--text) !important; font-size: .85rem !important; }

/* ── Tab bar ─────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: .88rem !important;
}

/* ── Spinner ─────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar brand + upload panel
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-icon">🚦</div>
            <p class="brand-title">ViolaTraffic</p>
            <p class="brand-sub">Smart Violation Detection System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-heading">📁 Upload Media</p>', unsafe_allow_html=True)
    file_type = st.radio("Media Type", ["Image", "Video"], horizontal=True, label_visibility="collapsed")

    accepted = (
        ["jpg", "jpeg", "png", "webp"]
        if file_type == "Image"
        else ["mp4", "mov", "avi"]
    )
    uploaded_file = st.file_uploader(
        f"Drop your {file_type.lower()} here",
        type=accepted,
        label_visibility="collapsed",
    )

    st.markdown('<p class="section-heading">⚙️ Options</p>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)

    run_btn = st.button("🔍  Run Detection", use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:.72rem; color:#475569; text-align:center;">© 2026 ViolaTraffic · Powered by YOLOv8</p>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <span class="ph-icon">🚨</span>
        <div>
            <p class="ph-title">Traffic Violation Detector</p>
            <p class="ph-sub">AI-powered helmet & triple-riding enforcement · Backed by YOLOv8</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_detect, tab_history = st.tabs(["🔍  Detection", "📋  History"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    if uploaded_file is None:
        st.markdown(
            """
            <div class="upload-hint">
                <div style="font-size:2.5rem;margin-bottom:.5rem">📷</div>
                <strong>No file selected</strong><br>
                Use the sidebar to upload an image or video, then click <em>Run Detection</em>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ── Image flow ────────────────────────────────────────────────────────
        if file_type == "Image":
            col_orig, col_ann = st.columns(2, gap="large")

            with col_orig:
                st.markdown('<p class="section-heading">📷 Original Image</p>', unsafe_allow_html=True)
                orig_placeholder = st.empty()
                orig_placeholder.image(Image.open(uploaded_file), use_container_width=True)

            with col_ann:
                st.markdown('<p class="section-heading">✅ Detection Result</p>', unsafe_allow_html=True)
                res_placeholder = st.empty()
                # Initial placeholder to keep layout steady
                res_placeholder.markdown(
                    """<div class="upload-hint" style="height:300px; display:flex; align-items:center; justify-content:center;">
                    Waiting for analysis...</div>""", 
                    unsafe_allow_html=True
                )

            if run_btn:
                with st.spinner("🔍 Analyzing image with YOLOv8…"):
                    uploaded_file.seek(0)
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    try:
                        r = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                    except requests.exceptions.ConnectionError:
                        st.error("⚠️ Could not reach the backend. Make sure it is running on port 8000.")
                        st.stop()

                if r.status_code == 200:
                    data = r.json()
                    mc_data = data.get("motorcycles", [])

                    img_bytes = base64.b64decode(data["image_base64"])
                    img_arr   = np.frombuffer(img_bytes, np.uint8)
                    result_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    res_placeholder.image(result_img, use_container_width=True)

                    # ── Metric cards ─────────────────────────────────────────
                    st.markdown('<p class="section-heading">📊 Detection Summary</p>', unsafe_allow_html=True)
                    total_bikes  = len(mc_data)
                    violations   = sum(1 for m in mc_data if m["violation"] != "Safe")
                    # Count individual riders who have helmets across all bikes
                    safe_riders  = sum(
                        sum(1 for h in m["helmets"] if h is True)
                        for m in mc_data
                    )
                    triple_ride  = sum(1 for m in mc_data if "Triple" in m.get("violation", ""))

                    c1, c2, c3, c4 = st.columns(4)
                    for col, label, val, color, icon in [
                        (c1, "Total Bikes",    total_bikes,  "blue",  "🏍️"),
                        (c2, "Violations",      violations,  "red",   "🚨"),
                        (c3, "Safe Riders",     safe_riders, "green", "✅"),
                        (c4, "Triple Riding",   triple_ride, "warn",  "⚠️"),
                    ]:
                        with col:
                            st.markdown(
                                f"""<div class="stat-card {color}">
                                    <div class="sc-label">{icon} {label}</div>
                                    <div class="sc-value">{val}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                    # ── Detailed log ─────────────────────────────────────────
                    st.markdown('<p class="section-heading">📋 Detailed Logs</p>', unsafe_allow_html=True)
                    if total_bikes > 0:
                        df = pd.DataFrame(mc_data)
                        df["helmets"] = df["helmets"].apply(
                            lambda x: ", ".join(["✅ Yes" if v else "❌ No" for v in x])
                        )
                        df["violation"] = df["violation"].apply(
                            lambda v: f"🟢 {v}" if v == "Safe" else f"🔴 {v}"
                        )
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No motorcycles detected in this frame.")
                else:
                    st.error(f"Backend returned error {r.status_code}.")

        # ── Video flow ────────────────────────────────────────────────────────
        elif file_type == "Video":
            st.markdown('<p class="section-heading">🎬 Uploaded Video</p>', unsafe_allow_html=True)
            st.video(uploaded_file)

            if run_btn:
                with st.spinner("🎞️ Processing video frame-by-frame… this may take a moment."):
                    uploaded_file.seek(0)
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")
                    }
                    try:
                        r = requests.post(f"{API_URL}/predict", files=files, timeout=300)
                    except requests.exceptions.ConnectionError:
                        st.error("⚠️ Could not reach the backend. Make sure it is running on port 8000.")
                        st.stop()

                if r.status_code == 200:
                    output_path = "output_processed.mp4"
                    with open(output_path, "wb") as f:
                        f.write(r.content)

                    st.success("🎉 Video processed successfully!")
                    st.markdown('<p class="section-heading">🎬 Annotated Output</p>', unsafe_allow_html=True)
                    st.video(output_path)

                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="⬇️  Download Processed Video",
                            data=file,
                            file_name="processed_violation.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )
                else:
                    st.error(f"Error processing video. Status: {r.status_code}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – History
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<p class="section-heading">📋 Detection History</p>', unsafe_allow_html=True)

    col_refresh, col_clear, col_spacer = st.columns([1, 1, 6])

    with col_refresh:
        refresh_btn = st.button("🔄  Refresh", use_container_width=True)
    with col_clear:
        clear_btn = st.button("🗑️  Clear All", use_container_width=True)

    if clear_btn:
        try:
            cr = requests.delete(f"{API_URL}/history/clear", timeout=10)
            if cr.status_code == 200:
                st.success("History cleared.")
            else:
                st.error("Failed to clear history.")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Backend unreachable.")

    # Fetch history
    try:
        hr = requests.get(f"{API_URL}/history", timeout=10)
        history = hr.json().get("history", []) if hr.status_code == 200 else []
    except requests.exceptions.ConnectionError:
        history = []
        st.warning("⚠️ Backend is not reachable. Start the API server to load history.")

    if not history:
        st.markdown(
            """
            <div class="upload-hint">
                <div style="font-size:2.5rem;margin-bottom:.5rem">🗂️</div>
                <strong>No records yet</strong><br>
                Run a detection first and results will appear here automatically.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ── Summary bar ───────────────────────────────────────────────────────
        total_runs       = len(history)
        total_violations = sum(r["violations"] for r in history)
        total_bikes_all  = sum(r["total_bikes"] for r in history)
        image_runs       = sum(1 for r in history if r["media_type"] == "image")

        hc1, hc2, hc3, hc4 = st.columns(4)
        for col, label, val, color, icon in [
            (hc1, "Total Scans",     total_runs,       "blue",  "📊"),
            (hc2, "Total Violations",total_violations,  "red",   "🚨"),
            (hc3, "Bikes Detected",  total_bikes_all,  "green", "🏍️"),
            (hc4, "Image Scans",     image_runs,       "warn",  "📷"),
        ]:
            with col:
                st.markdown(
                    f"""<div class="stat-card {color}">
                        <div class="sc-label">{icon} {label}</div>
                        <div class="sc-value">{val}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown('<p class="section-heading">📜 Records</p>', unsafe_allow_html=True)

        for idx, record in enumerate(history):
            chip_cls  = "img" if record["media_type"] == "image" else "vid"
            chip_icon = "🖼️" if record["media_type"] == "image" else "🎬"
            has_viol  = record["violations"] > 0
            badge_cls = "danger" if has_viol else "safe"
            badge_lbl = f"🔴 {record['violations']} Violation(s)" if has_viol else "🟢 Safe"

            st.markdown(
                f"""
                <div class="hist-row">
                    <div>
                        <div class="hr-name">{record['filename']}</div>
                        <div class="hr-ts">🕐 {record['timestamp']}</div>
                    </div>
                    <span class="hr-chip {chip_cls}">{chip_icon} {record['media_type'].capitalize()}</span>
                    <div style="text-align:center;">
                        <div style="font-size:.72rem;color:#64748b;">Bikes</div>
                        <div style="font-size:1.3rem;font-weight:800;">{record['total_bikes']}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:.72rem;color:#64748b;">Safe</div>
                        <div style="font-size:1.3rem;font-weight:800;color:#4ade80;">{record['safe_riders']}</div>
                    </div>
                    <span class="vbadge {badge_cls}">{badge_lbl}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Expandable details per record
            with st.expander(f"View details – {record['filename']}", expanded=False):
                details = record.get("details", [])
                if details:
                    df_d = pd.DataFrame(details)
                    df_d["helmets"] = df_d["helmets"].apply(
                        lambda x: ", ".join(["✅ Yes" if v else "❌ No" for v in x])
                        if isinstance(x, list) else str(x)
                    )
                    df_d["violation"] = df_d["violation"].apply(
                        lambda v: f"🟢 {v}" if v == "Safe" else f"🔴 {v}"
                    )
                    st.dataframe(df_d, use_container_width=True, hide_index=True)
                else:
                    st.info("No motorcycle data for this record.")
