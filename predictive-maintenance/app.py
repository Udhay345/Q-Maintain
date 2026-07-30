import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import datetime
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

from utils.preprocessing import clean_columns, preprocess_features, REQUIRED_FEATURES
from utils.prediction import predict_health_status
from utils.phyphox_client import PhyphoxClient
from utils.feature_engineering import extract_features
from utils.rul import estimate_health_and_rul

st.set_page_config(
    page_title="Live Machine Health Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

STATUS_LABELS = {0: "GOOD", 1: "BAD", 2: "CRITICAL"}
STATUS_COLORS = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444"}
STATUS_EMOJI = {0: "🟢", 1: "🟠", 2: "🔴"}
MAX_ANALYSIS_HISTORY = 60  # last 60 seconds of live analysis

st.markdown("""
<style>
    body { color: #222; background-color: #f7f8fa; }
    .header-container {
        padding: 1.25rem 0 0.75rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.25rem;
    }
    .main-title {
        font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em;
        color: #111; margin: 0;
    }
    .sub-title { font-size: 0.95rem; color: #666; margin-top: 0.15rem; }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #111;
        margin: 1.1rem 0 0.85rem 0; border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.35rem;
    }
    .live-banner {
        display: flex; align-items: center; gap: 0.6rem;
        font-size: 0.85rem; font-weight: 600; color: #374151;
        margin-bottom: 0.75rem;
    }
    .pulse-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: #10b981; box-shadow: 0 0 0 0 rgba(16,185,129,0.7);
        animation: pulse 1.2s infinite;
    }
    .pulse-dot.off { background: #9ca3af; animation: none; box-shadow: none; }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.55); }
        70% { box-shadow: 0 0 0 10px rgba(16,185,129,0); }
        100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
    }
    .result-card {
        background: #fff; border: 1px solid #e5e7eb; border-radius: 14px;
        padding: 1.5rem 1.25rem; text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    }
    .result-card.good { border-left: 6px solid #10b981; background: linear-gradient(135deg,#ecfdf5 0%,#fff 55%); }
    .result-card.bad { border-left: 6px solid #f59e0b; background: linear-gradient(135deg,#fffbeb 0%,#fff 55%); }
    .result-card.critical { border-left: 6px solid #ef4444; background: linear-gradient(135deg,#fef2f2 0%,#fff 55%); }
    .result-header {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.14em;
        color: #6b7280; text-transform: uppercase; margin-bottom: 0.35rem;
    }
    .status-val { font-size: 2.6rem; font-weight: 800; margin: 0.35rem 0; letter-spacing: -0.03em; }
    .status-good { color: #10b981; }
    .status-bad { color: #f59e0b; }
    .status-critical { color: #ef4444; }
    .confidence-text { font-size: 0.95rem; font-weight: 600; color: #374151; }
    .tick-text { font-size: 0.8rem; color: #6b7280; margin-top: 0.35rem; }
    .pred-card {
        background: #fff; border: 1px solid #e5e7eb; border-radius: 14px;
        padding: 1.1rem 1rem 1.2rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.15s ease;
        min-height: 148px;
    }
    .pred-card.active-good {
        border: 2px solid #10b981; background: linear-gradient(160deg,#ecfdf5 0%,#fff 70%);
        box-shadow: 0 8px 20px rgba(16,185,129,0.18); transform: translateY(-2px);
    }
    .pred-card.active-bad {
        border: 2px solid #f59e0b; background: linear-gradient(160deg,#fffbeb 0%,#fff 70%);
        box-shadow: 0 8px 20px rgba(245,158,11,0.18); transform: translateY(-2px);
    }
    .pred-card.active-critical {
        border: 2px solid #ef4444; background: linear-gradient(160deg,#fef2f2 0%,#fff 70%);
        box-shadow: 0 8px 20px rgba(239,68,68,0.18); transform: translateY(-2px);
    }
    .pred-card .pred-name { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; color: #6b7280; text-transform: uppercase; }
    .pred-card .pred-emoji { font-size: 1.6rem; margin: 0.35rem 0 0.15rem; }
    .pred-card .pred-pct { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; }
    .pred-card .pred-bar {
        height: 8px; border-radius: 999px; background: #e5e7eb; margin-top: 0.75rem; overflow: hidden;
    }
    .pred-card .pred-bar > span { display: block; height: 100%; border-radius: 999px; }
    .pred-card .pred-tag {
        display: inline-block; margin-top: 0.55rem; font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase; padding: 0.2rem 0.55rem; border-radius: 999px;
        background: #111; color: #fff;
    }
    .mini-metric {
        background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 0.9rem 1rem; text-align: center;
    }
    .mini-metric .label { font-size: 0.72rem; font-weight: 700; color: #6b7280; letter-spacing: 0.08em; text-transform: uppercase; }
    .mini-metric .value { font-size: 1.45rem; font-weight: 800; color: #111; margin-top: 0.25rem; }
    .sensor-card {
        background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 1rem; margin-bottom: 0.75rem;
    }
    .sensor-title {
        font-size: 0.8rem; font-weight: 700; color: #374151;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
PREPROCESSING_PATH = os.path.join(MODEL_DIR, "preprocessing.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Session state ----------
defaults = {
    "connected": False,
    "live_active": False,
    "phyphox_client": None,
    "buffer_mapping": {},
    "last_buffer_counts": {},
    "live_buffer": pd.DataFrame(columns=REQUIRED_FEATURES),
    "history_log": [],
    "analysis_history": [],
    "consecutive_failures": 0,
    "reset_buffers": True,
    "last_analysis_time": None,
    "latest_phyphox_json": None,
    "available_buffers": [],
    "poll_debug": "",
    "phyphox_measuring": None,
    "autostart_attempted": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
# Drop legacy demo mode if present from older sessions
st.session_state.demo_mode = False

def load_model_from_disk():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        loaded = joblib.load(MODEL_PATH)
        prep = joblib.load(PREPROCESSING_PATH) if os.path.exists(PREPROCESSING_PATH) else {"medians": {}, "metrics": None}
        if isinstance(loaded, dict):
            return {
                "model": loaded.get("model", loaded),
                "medians": loaded.get("medians", prep.get("medians", {})),
                "metrics": loaded.get("metrics", prep.get("metrics")),
                "class_centroids": loaded.get("class_centroids", prep.get("class_centroids", {})),
                "mtime": os.path.getmtime(MODEL_PATH),
            }
        return {
            "model": loaded,
            "medians": prep.get("medians", {}),
            "metrics": prep.get("metrics"),
            "class_centroids": prep.get("class_centroids", {}),
            "mtime": os.path.getmtime(MODEL_PATH),
        }
    except Exception:
        return None


# Always pick up freshly trained model artifacts
_disk_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
_need_reload = (
    "model_data" not in st.session_state
    or st.session_state.model_data is None
    or st.session_state.model_data.get("mtime") != _disk_mtime
)
if _need_reload:
    st.session_state.model_data = load_model_from_disk()


# Real class profiles from training (fallback if preprocessing lacks centroids)
CLASS_PROFILES = {
    0: {  # GOOD
        "Acceleration x (m/s^2)": -1.1178, "Acceleration y (m/s^2)": -0.0631, "Acceleration z (m/s^2)": 9.3425,
        "Gyroscope x (rad/s)": 0.0, "Gyroscope y (rad/s)": -0.0001, "Gyroscope z (rad/s)": 0.0,
        "Linear Acceleration x (m/s^2)": 0.0491, "Linear Acceleration y (m/s^2)": 0.0014, "Linear Acceleration z (m/s^2)": -0.3794,
        "Magnetic field x (µT)": 26.5313, "Magnetic field y (µT)": 32.3625, "Magnetic field z (µT)": -17.0063,
        "noise": {"acc": 0.04, "gyro": 0.02, "lin": 0.01, "mag": 0.4},
    },
    1: {  # BAD
        "Acceleration x (m/s^2)": 5.5815, "Acceleration y (m/s^2)": -4.9632, "Acceleration z (m/s^2)": 1.2650,
        "Gyroscope x (rad/s)": -0.0162, "Gyroscope y (rad/s)": 0.0274, "Gyroscope z (rad/s)": -0.0132,
        "Linear Acceleration x (m/s^2)": -0.1401, "Linear Acceleration y (m/s^2)": -0.2233, "Linear Acceleration z (m/s^2)": 0.0137,
        "Magnetic field x (µT)": 1.0500, "Magnetic field y (µT)": 1.4438, "Magnetic field z (µT)": -5.9250,
        "noise": {"acc": 1.2, "gyro": 0.35, "lin": 0.45, "mag": 1.0},
    },
    2: {  # CRITICAL
        "Acceleration x (m/s^2)": 9.2566, "Acceleration y (m/s^2)": -8.4851, "Acceleration z (m/s^2)": 7.7156,
        "Gyroscope x (rad/s)": -0.1634, "Gyroscope y (rad/s)": 0.2786, "Gyroscope z (rad/s)": -0.4423,
        "Linear Acceleration x (m/s^2)": 3.6949, "Linear Acceleration y (m/s^2)": -2.6257, "Linear Acceleration z (m/s^2)": 2.6950,
        "Magnetic field x (µT)": 8.5875, "Magnetic field y (µT)": -13.9875, "Magnetic field z (µT)": -14.5688,
        "noise": {"acc": 4.5, "gyro": 1.8, "lin": 2.2, "mag": 1.5},
    },
}


# Exact buffer names from phyphox/smartphone_predictive_maintenance.phyphox
PHYPOX_EXACT_BUFFERS = {
    "Acceleration x (m/s^2)": "acc_x",
    "Acceleration y (m/s^2)": "acc_y",
    "Acceleration z (m/s^2)": "acc_z",
    "Gyroscope x (rad/s)": "gyro_x",
    "Gyroscope y (rad/s)": "gyro_y",
    "Gyroscope z (rad/s)": "gyro_z",
    "Linear Acceleration x (m/s^2)": "lin_acc_x",
    "Linear Acceleration y (m/s^2)": "lin_acc_y",
    "Linear Acceleration z (m/s^2)": "lin_acc_z",
    "Magnetic field x (µT)": "mag_x",
    "Magnetic field y (µT)": "mag_y",
    "Magnetic field z (µT)": "mag_z",
}


def map_phyphox_buffers(available_buffers):
    """Map required features to Phyphox buffer names. Prefer exact custom-experiment names."""
    available = [str(b) for b in available_buffers]
    available_lower = {b.lower(): b for b in available}
    mapping = {}

    def find_exact_or_fuzzy(exact_name, must_include, must_exclude=None):
        must_exclude = must_exclude or []
        if exact_name.lower() in available_lower:
            return available_lower[exact_name.lower()]
        if exact_name in available:
            return exact_name
        for buf in available:
            low = buf.lower()
            if all(k in low for k in must_include) and not any(x in low for x in must_exclude):
                return buf
        return exact_name

    mapping["Acceleration x (m/s^2)"] = find_exact_or_fuzzy("acc_x", ["acc", "x"], ["lin", "linear"])
    mapping["Acceleration y (m/s^2)"] = find_exact_or_fuzzy("acc_y", ["acc", "y"], ["lin", "linear"])
    mapping["Acceleration z (m/s^2)"] = find_exact_or_fuzzy("acc_z", ["acc", "z"], ["lin", "linear"])
    mapping["Gyroscope x (rad/s)"] = find_exact_or_fuzzy("gyro_x", ["gyro", "x"])
    mapping["Gyroscope y (rad/s)"] = find_exact_or_fuzzy("gyro_y", ["gyro", "y"])
    mapping["Gyroscope z (rad/s)"] = find_exact_or_fuzzy("gyro_z", ["gyro", "z"])
    mapping["Linear Acceleration x (m/s^2)"] = find_exact_or_fuzzy("lin_acc_x", ["lin", "x"])
    mapping["Linear Acceleration y (m/s^2)"] = find_exact_or_fuzzy("lin_acc_y", ["lin", "y"])
    mapping["Linear Acceleration z (m/s^2)"] = find_exact_or_fuzzy("lin_acc_z", ["lin", "z"])
    mapping["Magnetic field x (µT)"] = find_exact_or_fuzzy("mag_x", ["mag", "x"])
    mapping["Magnetic field y (µT)"] = find_exact_or_fuzzy("mag_y", ["mag", "y"])
    mapping["Magnetic field z (µT)"] = find_exact_or_fuzzy("mag_z", ["mag", "z"])
    return mapping


def _extract_buffer_values(buf_data):
    """Normalize Phyphox buffer JSON entry into a list of floats."""
    if buf_data is None:
        return []
    if isinstance(buf_data, list):
        out = []
        for v in buf_data:
            if v is None:
                continue
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(buf_data, dict):
        vals = buf_data.get("buffer", [])
        if vals is None:
            vals = []
        out = []
        for v in vals:
            if v is None:
                continue
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return [float(buf_data)]
    except (TypeError, ValueError):
        return []


def append_analysis_point(pred_class, pred_proba, health_score, estimated_rul_hours, trend, features):
    now = datetime.datetime.now()
    last = st.session_state.last_analysis_time
    if last is not None and (now - last).total_seconds() < 0.95:
        return False

    st.session_state.last_analysis_time = now
    point = {
        "time": now,
        "time_str": now.strftime("%H:%M:%S"),
        "status": int(pred_class),
        "status_label": STATUS_LABELS.get(int(pred_class), "UNKNOWN"),
        "confidence": float(pred_proba[pred_class]),
        "p_good": float(pred_proba[0]),
        "p_bad": float(pred_proba[1]),
        "p_critical": float(pred_proba[2]),
        "health": float(health_score),
        "rul_hours": float(estimated_rul_hours),
        "trend": trend,
        "acc_rms": float(features.get("acc_rms", 0.0)),
        "gyro_rms": float(features.get("gyro_rms", 0.0)),
    }
    st.session_state.analysis_history.append(point)
    if len(st.session_state.analysis_history) > MAX_ANALYSIS_HISTORY:
        st.session_state.analysis_history = st.session_state.analysis_history[-MAX_ANALYSIS_HISTORY:]

    st.session_state.history_log.append({
        "Time": point["time_str"],
        "Status": point["status_label"],
        "Confidence": f"{point['confidence']:.0%}",
        "Health": f"{point['health']:.0f}%",
        "Accel RMS": f"{point['acc_rms']:.2f}",
        "Gyro RMS": f"{point['gyro_rms']:.3f}",
    })
    if len(st.session_state.history_log) > 12:
        st.session_state.history_log = st.session_state.history_log[-12:]
    return True


def analysis_history_df():
    if not st.session_state.analysis_history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.analysis_history)


def render_prediction_cards(pred_class, pred_proba, tick_time=None):
    """Neat Good / Bad / Critical probability cards with winner highlighted."""
    names = ["GOOD", "BAD", "CRITICAL"]
    keys = ["good", "bad", "critical"]
    colors = [STATUS_COLORS[0], STATUS_COLORS[1], STATUS_COLORS[2]]
    emojis = [STATUS_EMOJI[0], STATUS_EMOJI[1], STATUS_EMOJI[2]]

    style_map = {0: ("good", "status-good"), 1: ("bad", "status-bad"), 2: ("critical", "status-critical")}
    card_cls, text_cls = style_map.get(pred_class, ("good", "status-good"))
    conf = float(pred_proba[pred_class])
    tick = tick_time or datetime.datetime.now().strftime("%H:%M:%S")

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="result-header">Predicted Health Status</div>
        <div class="status-val {text_cls}">{emojis[pred_class]} {names[pred_class]}</div>
        <div class="confidence-text">{conf:.1%} confidence</div>
        <div class="tick-text">Model prediction · refreshed {tick}</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for i, col in enumerate(cols):
        active = "active-" + keys[i] if i == pred_class else ""
        pct = float(pred_proba[i]) * 100.0
        width = max(2.0, min(100.0, pct))
        tag = '<div class="pred-tag">SELECTED</div>' if i == pred_class else ""
        with col:
            st.markdown(f"""
            <div class="pred-card {active}">
                <div class="pred-name">{names[i]}</div>
                <div class="pred-emoji">{emojis[i]}</div>
                <div class="pred-pct" style="color:{colors[i]}">{pct:.1f}%</div>
                <div class="pred-bar"><span style="width:{width:.1f}%; background:{colors[i]};"></span></div>
                {tag}
            </div>
            """, unsafe_allow_html=True)


def render_live_dashboard(pred_class, pred_proba, health_score, estimated_rul_hours, trend, latest_row):
    hist = analysis_history_df()
    tick_time = datetime.datetime.now().strftime("%H:%M:%S")

    is_live = st.session_state.live_active
    source = st.session_state.get("phyphox_url", "Phyphox HTTP")
    pulse_cls = "pulse-dot" if is_live else "pulse-dot off"
    n_samples = len(st.session_state.live_buffer)
    st.markdown(f"""
    <div class="live-banner">
        <div class="{pulse_cls}"></div>
        {"LIVE from Phyphox JSON" if is_live else "IDLE"} · {source} · buffer {n_samples} samples
    </div>
    """, unsafe_allow_html=True)

    render_prediction_cards(pred_class, pred_proba, tick_time)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="mini-metric">
            <div class="label">Health Score</div>
            <div class="value" style="color:{STATUS_COLORS.get(pred_class, '#111')}">{health_score:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(min(1.0, max(0.0, health_score / 100.0))))
    with m2:
        rul_txt = f"{estimated_rul_hours:.0f} hrs" if estimated_rul_hours < 48 else f"{estimated_rul_hours/24:.1f} d"
        st.markdown(f"""
        <div class="mini-metric">
            <div class="label">Est. RUL</div>
            <div class="value">{rul_txt}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        trend_color = "#10b981" if trend == "Stable" else ("#f59e0b" if trend == "Degrading" else "#ef4444")
        st.markdown(f"""
        <div class="mini-metric">
            <div class="label">Trend</div>
            <div class="value" style="color:{trend_color}; font-size:1.15rem;">{trend}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Live Analysis Visualizations</div>', unsafe_allow_html=True)

    if not hist.empty:
        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.caption("Health score over time (last 60 s)")
            health_chart = hist.set_index("time_str")[["health"]].rename(columns={"health": "Health %"})
            st.line_chart(health_chart, height=220, color="#2563eb")

            st.caption("Status timeline (0=Good · 1=Bad · 2=Critical)")
            status_chart = hist.set_index("time_str")[["status"]].rename(columns={"status": "Status code"})
            st.area_chart(status_chart, height=180, color="#f59e0b")

        with chart_right:
            st.caption("Class confidence over time")
            conf_chart = hist.set_index("time_str")[["p_good", "p_bad", "p_critical"]].rename(
                columns={"p_good": "Good", "p_bad": "Bad", "p_critical": "Critical"}
            )
            st.area_chart(conf_chart, height=220)

            st.caption("Vibration indicators (RMS)")
            rms_chart = hist.set_index("time_str")[["acc_rms", "gyro_rms"]].rename(
                columns={"acc_rms": "Accel RMS", "gyro_rms": "Gyro RMS"}
            )
            st.line_chart(rms_chart, height=180)
    else:
        st.info("Collecting first analysis tick… charts appear after 1 second.")

    # Sensor stream charts
    if not st.session_state.live_buffer.empty:
        st.markdown('<div class="section-title">Live Sensor Streams</div>', unsafe_allow_html=True)
        plot_df = st.session_state.live_buffer.iloc[-100:].reset_index(drop=True)
        g1, g2 = st.columns(2)
        with g1:
            st.caption("Accelerometer (m/s²)")
            st.line_chart(
                plot_df[["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]],
                height=150,
            )
            st.caption("Gyroscope (rad/s)")
            st.line_chart(
                plot_df[["Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)"]],
                height=150,
            )
        with g2:
            st.caption("Linear acceleration (m/s²)")
            st.line_chart(
                plot_df[[
                    "Linear Acceleration x (m/s^2)",
                    "Linear Acceleration y (m/s^2)",
                    "Linear Acceleration z (m/s^2)",
                ]],
                height=150,
            )
            st.caption("Magnetometer (µT)")
            st.line_chart(
                plot_df[["Magnetic field x (µT)", "Magnetic field y (µT)", "Magnetic field z (µT)"]],
                height=150,
            )

        vals = latest_row.iloc[0]
        st.caption(
            f"Latest · Acc ({vals['Acceleration x (m/s^2)']:.2f}, "
            f"{vals['Acceleration y (m/s^2)']:.2f}, {vals['Acceleration z (m/s^2)']:.2f}) · "
            f"Gyro ({vals['Gyroscope x (rad/s)']:.3f}, {vals['Gyroscope y (rad/s)']:.3f}, "
            f"{vals['Gyroscope z (rad/s)']:.3f})"
        )

    if st.session_state.history_log:
        st.markdown('<div class="section-title">Analysis Log (1 tick / second)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history_log).iloc[::-1], use_container_width=True, hide_index=True)

    if st.session_state.latest_phyphox_json is not None:
        st.markdown('<div class="section-title">Phyphox Sensor JSON (live /get)</div>', unsafe_allow_html=True)
        raw = st.session_state.latest_phyphox_json
        preview = {"source": st.session_state.get("phyphox_url"), "buffers": {}}
        buf = raw.get("buffer", {}) if isinstance(raw, dict) else {}
        for name, entry in list(buf.items())[:12]:
            vals = _extract_buffer_values(entry)
            preview["buffers"][name] = {
                "n_values": len(vals),
                "latest": vals[-1] if vals else None,
                "updateMode": entry.get("updateMode") if isinstance(entry, dict) else None,
            }
        st.json(preview)
        with st.expander("Full raw Phyphox JSON", expanded=False):
            st.code(json.dumps(raw, indent=2)[:6000], language="json")


def poll_phyphox_into_buffer():
    """
    Pull sensor JSON from Phyphox using /get?buf=full (correct API).
    Rebuilds live_buffer from the phone's latest window so charts/predictions always show.
    """
    client = st.session_state.phyphox_client
    if client is None:
        st.session_state.poll_debug = "No Phyphox client — click Connect first"
        return False

    mapping = st.session_state.buffer_mapping or dict(PHYPOX_EXACT_BUFFERS)
    buf_names = list(dict.fromkeys(mapping.values()))

    # Always request FULL buffers — Phyphox does NOT use index/count thresholds
    data = client.get_full(buf_names)
    if data is None:
        data = client.get_latest(buf_names)
    if data is None:
        st.session_state.consecutive_failures += 1
        st.session_state.poll_debug = f"HTTP fetch failed: {client.last_error}"
        return False

    st.session_state.latest_phyphox_json = data
    st.session_state.consecutive_failures = 0
    measuring = client.is_measuring(data)
    st.session_state.phyphox_measuring = measuring

    if measuring is False and not st.session_state.get("autostart_attempted"):
        st.session_state.autostart_attempted = True
        client.start()
        data = client.get_full(buf_names) or data
        st.session_state.latest_phyphox_json = data
        measuring = client.is_measuring(data)
        st.session_state.phyphox_measuring = measuring

    if "buffer" not in data or not isinstance(data["buffer"], dict):
        st.session_state.poll_debug = f"JSON has no 'buffer' key. Keys={list(data.keys())}"
        return False

    # Refresh mapping from actual keys if needed
    actual_keys = list(data["buffer"].keys())
    if actual_keys and (
        not st.session_state.available_buffers
        or any(b not in actual_keys for b in buf_names)
    ):
        st.session_state.available_buffers = actual_keys
        st.session_state.buffer_mapping = map_phyphox_buffers(actual_keys)
        mapping = st.session_state.buffer_mapping
        buf_names = list(dict.fromkeys(mapping.values()))

    series = {}
    lengths = {}
    missing = []
    for feat in REQUIRED_FEATURES:
        buf = mapping.get(feat, PHYPOX_EXACT_BUFFERS.get(feat))
        if buf not in data["buffer"]:
            missing.append(str(buf))
            series[feat] = []
            lengths[feat] = 0
            continue
        vals = _extract_buffer_values(data["buffer"][buf])
        series[feat] = vals
        lengths[feat] = len(vals)

    nonempty = [n for n in lengths.values() if n > 0]
    if not nonempty:
        row = {}
        ok = True
        for feat in REQUIRED_FEATURES:
            vals = series[feat]
            if not vals:
                ok = False
                break
            row[feat] = vals[-1]
        if ok:
            new_df = pd.DataFrame([row])
            st.session_state.live_buffer = pd.concat(
                [st.session_state.live_buffer, new_df], ignore_index=True
            ).iloc[-300:]
            st.session_state.poll_debug = (
                f"OK latest-value mode · measuring={measuring} · "
                f"buffer_rows={len(st.session_state.live_buffer)}"
            )
            return True
        st.session_state.poll_debug = (
            f"No sensor values yet · measuring={measuring} · missing={missing[:6]} · "
            f"available={actual_keys[:12]}"
        )
        return False

    min_len = min(nonempty)
    aligned = {}
    for feat in REQUIRED_FEATURES:
        vals = series[feat]
        if not vals:
            aligned[feat] = [0.0] * min_len
        else:
            aligned[feat] = vals[-min_len:]

    window = min(200, min_len)
    new_df = pd.DataFrame({feat: aligned[feat][-window:] for feat in REQUIRED_FEATURES})
    st.session_state.live_buffer = new_df
    st.session_state.poll_debug = (
        f"OK full-buffer mode · measuring={measuring} · "
        f"phone_samples={min_len} · chart_rows={len(new_df)} · missing={missing[:4]}"
    )
    return True


def run_live_analysis_tick():
    """Compute Good/Bad/Critical + RUL from latest buffer and record 1 Hz history."""
    if st.session_state.live_buffer.empty:
        return None

    latest_row = st.session_state.live_buffer.iloc[[-1]]
    # Average over recent samples for stabler live class cards
    window_rows = st.session_state.live_buffer.iloc[-15:]
    medians = None
    if st.session_state.model_data and "medians" in st.session_state.model_data:
        medians = st.session_state.model_data["medians"]
    X_prep = preprocess_features(window_rows, medians=medians)

    pred_proba = np.array([1.0, 0.0, 0.0])
    pred_class = 0
    if st.session_state.model_data is not None:
        try:
            pred_class, pred_proba = predict_health_status(st.session_state.model_data["model"], X_prep)
        except Exception as ex:
            st.error(f"Live prediction error: {ex}")
            return None
    else:
        st.warning("No trained model — click Retrain on final_phone_sensor.csv in the sidebar.")

    df_window = st.session_state.live_buffer.iloc[-100:]
    features = extract_features(df_window)
    health_score, estimated_rul_hours, trend = estimate_health_and_rul(features, pred_proba)
    append_analysis_point(pred_class, pred_proba, health_score, estimated_rul_hours, trend, features)
    return pred_class, pred_proba, health_score, estimated_rul_hours, trend, latest_row


# ===================== SIDEBAR: connection + training =====================
with st.sidebar:
    st.markdown("### Phyphox Live")
    st.caption("Uses real sensor JSON from the phone HTTP URL — no random/demo data.")

    phyphox_url = st.text_input(
        "Phyphox HTTP URL",
        value=st.session_state.get("phyphox_url", "http://192.168.1.100:8080"),
        help="Enable Remote Access in Phyphox and paste the URL shown on the phone.",
    )
    st.session_state.phyphox_url = phyphox_url

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Connect", use_container_width=True):
            client = PhyphoxClient(phyphox_url)
            success, res = client.connect()
            if success:
                st.session_state.connected = True
                st.session_state.phyphox_client = client
                names = client.list_buffer_names(res) if isinstance(res, dict) else []
                if not names:
                    # probe /get for buffer keys
                    probe = client.get_data()
                    names = client.list_buffer_names(probe) if probe else []
                st.session_state.available_buffers = names
                st.session_state.buffer_mapping = map_phyphox_buffers(names or list(PHYPOX_EXACT_BUFFERS.values()))
                st.session_state.reset_buffers = True
                st.session_state.consecutive_failures = 0
                st.session_state.latest_phyphox_json = res if isinstance(res, dict) else None
                st.success(f"Connected · {len(st.session_state.buffer_mapping)} buffers mapped")
            else:
                st.session_state.connected = False
                st.error(f"Connection failed: {res}")
    with c2:
        if st.button("Disconnect", use_container_width=True, disabled=not st.session_state.connected):
            if st.session_state.live_active and st.session_state.phyphox_client:
                st.session_state.phyphox_client.stop()
            st.session_state.connected = False
            st.session_state.live_active = False
            st.session_state.phyphox_client = None
            st.session_state.last_buffer_counts = {}

    live_on = st.session_state.live_active
    start_disabled = (not st.session_state.connected) or live_on
    if st.button("▶ Start Live Analysis", type="primary", use_container_width=True, disabled=start_disabled):
        client = st.session_state.phyphox_client
        if client:
            client.start()
            st.session_state.live_active = True
            st.session_state.reset_buffers = True
            st.session_state.autostart_attempted = False
            st.session_state.analysis_history = []
            st.session_state.history_log = []
            st.session_state.last_analysis_time = None
            st.session_state.live_buffer = pd.DataFrame(columns=REQUIRED_FEATURES)
            st.rerun()
        else:
            st.error("Connect first")

    if st.button("⏹ Stop", use_container_width=True, disabled=not live_on):
        if st.session_state.phyphox_client:
            st.session_state.phyphox_client.stop()
        st.session_state.live_active = False
        st.rerun()

    if st.session_state.connected:
        st.markdown("<span style='color:#10b981;font-weight:700;'>● CONNECTED</span>", unsafe_allow_html=True)
        if st.session_state.available_buffers:
            st.caption("Buffers: " + ", ".join(st.session_state.available_buffers[:12]))
    else:
        st.markdown("<span style='color:#ef4444;font-weight:700;'>● DISCONNECTED</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Trained Model")
    if st.session_state.model_data and st.session_state.model_data.get("metrics"):
        m = st.session_state.model_data["metrics"]
        n_total = m.get("n_total", "?")
        st.success(f"Ready · Acc {m['accuracy']:.1%} · {n_total} samples")
        st.caption(f"F1 {m['f1']:.1%} · Precision {m['precision']:.1%} · Recall {m['recall']:.1%}")
    else:
        st.warning("No model loaded")

    if st.button("Retrain on final_phone_sensor.csv", use_container_width=True):
        with st.spinner("Training full XGBoost on final_phone_sensor.csv..."):
            try:
                from train_model import train_and_save
                acc = train_and_save()
                st.session_state.model_data = load_model_from_disk()
                st.success(f"Fully trained · Accuracy {acc:.1%}")
                st.rerun()
            except Exception as ex:
                st.error(str(ex))

    with st.expander("Train / Model (advanced)", expanded=False):
        default_dataset_paths = [
            "../predictive/final_phone_sensor.csv",
            "../final_phone_sensor.csv",
            "./final_phone_sensor.csv",
            "predictive/final_phone_sensor.csv",
        ]
        sample_path = next((p for p in default_dataset_paths if os.path.exists(p)), None)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if sample_path and st.button("Load sample dataset"):
            try:
                try:
                    st.session_state.df = pd.read_csv(sample_path, encoding="utf-8")
                except UnicodeDecodeError:
                    st.session_state.df = pd.read_csv(sample_path, encoding="latin1")
                st.success("Sample loaded")
            except Exception as e:
                st.error(str(e))

        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                try:
                    st.session_state.df = pd.read_csv(uploaded_file, encoding="utf-8")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    st.session_state.df = pd.read_csv(uploaded_file, encoding="latin1")
                st.success("Upload loaded")
            except Exception as e:
                st.error(str(e))

        if "df" in st.session_state and st.session_state.df is not None:
            df_clean = clean_columns(st.session_state.df.copy())
            st.write(f"Samples: {df_clean.shape[0]}")
            if st.button("Train XGBoost", type="primary"):
                with st.spinner("Training..."):
                    try:
                        missing_req = [f for f in REQUIRED_FEATURES if f not in df_clean.columns]
                        if missing_req:
                            st.error(f"Missing: {', '.join(missing_req)}")
                        elif "Status" not in df_clean.columns:
                            st.error("Missing Status column")
                        else:
                            X = df_clean[REQUIRED_FEATURES].copy()
                            y = df_clean["Status"].copy()
                            valid_idx = y.dropna().index
                            X, y = X.loc[valid_idx], y.loc[valid_idx]
                            for col in REQUIRED_FEATURES:
                                X[col] = pd.to_numeric(X[col], errors="coerce")
                            status_mapping = {"good": 0, "bad": 1, "critical": 2}
                            y_encoded = y.astype(str).str.lower().map(status_mapping)
                            valid_target = y_encoded.dropna().index
                            X = X.loc[valid_target]
                            y_encoded = y_encoded.loc[valid_target].astype(int)
                            if len(y_encoded) < 10:
                                st.error("Need >=10 samples")
                            else:
                                class_counts = y_encoded.value_counts()
                                stratify_y = y_encoded if (class_counts >= 2).all() else None
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y_encoded, test_size=0.20, stratify=stratify_y, random_state=42
                                )
                                medians = X_train.median().fillna(0.0)
                                X_train_imp = X_train.fillna(medians)
                                X_test_imp = X_test.fillna(medians)
                                model = XGBClassifier(
                                    n_estimators=400, max_depth=7, learning_rate=0.06,
                                    subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
                                    objective="multi:softprob", num_class=3,
                                    eval_metric="mlogloss", random_state=42, n_jobs=-1,
                                )
                                model.fit(X_train_imp, y_train)
                                y_pred = model.predict(X_test_imp)
                                acc = accuracy_score(y_test, y_pred)
                                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                                cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
                                joblib.dump(model, MODEL_PATH)
                                class_centroids = {
                                    name: X[y_encoded == label].median().to_dict()
                                    for label, name in [(0, "good"), (1, "bad"), (2, "critical")]
                                }
                                preprocessing_data = {
                                    "medians": medians.to_dict(),
                                    "metrics": {
                                        "accuracy": acc, "precision": prec,
                                        "recall": rec, "f1": f1, "cm": cm.tolist(),
                                        "n_train": int(len(y_train)),
                                        "n_test": int(len(y_test)),
                                        "n_total": int(len(y_encoded)),
                                    },
                                    "class_centroids": class_centroids,
                                }
                                joblib.dump(preprocessing_data, PREPROCESSING_PATH)
                                st.session_state.model_data = load_model_from_disk()
                                st.success(f"Trained · Accuracy {acc:.1%}")
                    except Exception as ex:
                        st.error(str(ex))

    with st.expander("Manual prediction", expanded=False):
        st.caption("Defaults match GOOD class medians from training data")
        g = CLASS_PROFILES[0]
        acc_x = st.number_input("Acc X", value=float(g["Acceleration x (m/s^2)"]), step=0.1, format="%.2f")
        acc_y = st.number_input("Acc Y", value=float(g["Acceleration y (m/s^2)"]), step=0.1, format="%.2f")
        acc_z = st.number_input("Acc Z", value=float(g["Acceleration z (m/s^2)"]), step=0.1, format="%.2f")
        gyro_x = st.number_input("Gyro X", value=float(g["Gyroscope x (rad/s)"]), step=0.01, format="%.3f")
        gyro_y = st.number_input("Gyro Y", value=float(g["Gyroscope y (rad/s)"]), step=0.01, format="%.3f")
        gyro_z = st.number_input("Gyro Z", value=float(g["Gyroscope z (rad/s)"]), step=0.01, format="%.3f")
        lin_x = st.number_input("Lin X", value=float(g["Linear Acceleration x (m/s^2)"]), step=0.05, format="%.3f")
        lin_y = st.number_input("Lin Y", value=float(g["Linear Acceleration y (m/s^2)"]), step=0.05, format="%.3f")
        lin_z = st.number_input("Lin Z", value=float(g["Linear Acceleration z (m/s^2)"]), step=0.05, format="%.3f")
        mag_x = st.number_input("Mag X", value=float(g["Magnetic field x (µT)"]), step=1.0, format="%.2f")
        mag_y = st.number_input("Mag Y", value=float(g["Magnetic field y (µT)"]), step=1.0, format="%.2f")
        mag_z = st.number_input("Mag Z", value=float(g["Magnetic field z (µT)"]), step=1.0, format="%.2f")
        if st.button("Predict", use_container_width=True):
            if not st.session_state.model_data:
                st.error("Train a model first")
            else:
                input_df = pd.DataFrame([{
                    "Acceleration x (m/s^2)": acc_x,
                    "Acceleration y (m/s^2)": acc_y,
                    "Acceleration z (m/s^2)": acc_z,
                    "Gyroscope x (rad/s)": gyro_x,
                    "Gyroscope y (rad/s)": gyro_y,
                    "Gyroscope z (rad/s)": gyro_z,
                    "Linear Acceleration x (m/s^2)": lin_x,
                    "Linear Acceleration y (m/s^2)": lin_y,
                    "Linear Acceleration z (m/s^2)": lin_z,
                    "Magnetic field x (µT)": mag_x,
                    "Magnetic field y (µT)": mag_y,
                    "Magnetic field z (µT)": mag_z,
                }])
                X_prep = preprocess_features(input_df, medians=st.session_state.model_data["medians"])
                pc, pp = predict_health_status(st.session_state.model_data["model"], X_prep)
                render_prediction_cards(pc, pp)


# ===================== MAIN: LIVE DASHBOARD =====================
_model_note = ""
if st.session_state.model_data and st.session_state.model_data.get("metrics"):
    _m = st.session_state.model_data["metrics"]
    _model_note = f" · XGBoost trained on {_m.get('n_total', 10422)} samples · accuracy {_m['accuracy']:.1%}"

st.markdown(f"""
<div class="header-container">
    <h1 class="main-title">Live Machine Health Dashboard</h1>
    <p class="sub-title">Predictions & charts from Phyphox HTTP sensor JSON{_model_note}</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Phyphox Connection</div>', unsafe_allow_html=True)
url_col, btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 1, 1])
with url_col:
    main_url = st.text_input(
        "HTTP URL from Phyphox Remote Access",
        value=st.session_state.get("phyphox_url", "http://192.168.1.100:8080"),
        label_visibility="collapsed",
        placeholder="http://192.168.x.x:8080",
        key="main_phyphox_url",
    )
    st.session_state.phyphox_url = main_url
with btn_col1:
    if st.button("Connect", key="main_connect", use_container_width=True):
        client = PhyphoxClient(main_url)
        success, res = client.connect()
        if success:
            st.session_state.connected = True
            st.session_state.phyphox_client = client
            names = client.list_buffer_names(res) if isinstance(res, dict) else []
            if not names:
                probe = client.get_data()
                names = client.list_buffer_names(probe) if probe else []
            st.session_state.available_buffers = names
            st.session_state.buffer_mapping = map_phyphox_buffers(names or list(PHYPOX_EXACT_BUFFERS.values()))
            st.session_state.reset_buffers = True
            st.session_state.consecutive_failures = 0
            st.session_state.latest_phyphox_json = res if isinstance(res, dict) else None
            st.success("Connected to Phyphox")
            st.rerun()
        else:
            st.session_state.connected = False
            st.error(f"Could not reach Phyphox at {main_url}")
with btn_col2:
    if st.button("Start Live", type="primary", key="main_start", use_container_width=True,
                 disabled=not st.session_state.connected or st.session_state.live_active):
        client = st.session_state.phyphox_client
        started = False
        if client:
            started = client.start()
            # Even if start returns false, experiment may already be measuring — still go live
            probe = client.get_full(list(st.session_state.buffer_mapping.values()) or list(PHYPOX_EXACT_BUFFERS.values()))
            measuring = client.is_measuring(probe) if probe else None
            if started or measuring or probe is not None:
                st.session_state.live_active = True
                st.session_state.reset_buffers = True
                st.session_state.autostart_attempted = False
                st.session_state.analysis_history = []
                st.session_state.history_log = []
                st.session_state.last_analysis_time = None
                st.session_state.live_buffer = pd.DataFrame(columns=REQUIRED_FEATURES)
                st.session_state.latest_phyphox_json = probe
                st.rerun()
            else:
                st.error(f"Cannot read sensors. {client.last_error or 'Open experiment + Allow remote access + press play on phone.'}")
        else:
            st.error("Connect first")
with btn_col3:
    if st.button("Stop", key="main_stop", use_container_width=True, disabled=not st.session_state.live_active):
        if st.session_state.phyphox_client:
            st.session_state.phyphox_client.stop()
        st.session_state.live_active = False
        st.rerun()

is_streaming = st.session_state.live_active

if is_streaming:
    got_data = poll_phyphox_into_buffer()
    debug = st.session_state.get("poll_debug", "")
    measuring = st.session_state.get("phyphox_measuring")

    status_cols = st.columns(3)
    status_cols[0].metric("Connection", "LIVE" if st.session_state.connected else "OFF")
    status_cols[1].metric("Phone measuring", "YES" if measuring else ("NO" if measuring is False else "?"))
    status_cols[2].metric("Samples in chart", len(st.session_state.live_buffer))
    if debug:
        st.caption(f"Poll: {debug}")

    if st.session_state.consecutive_failures > 5:
        st.error("No JSON from Phyphox — check the HTTP URL, same Wi-Fi, Remote Access enabled, experiment playing.")
    elif measuring is False:
        st.warning("Phyphox is connected but NOT measuring. Press the play/start button in the Phyphox app on your phone.")
    elif not got_data and st.session_state.live_buffer.empty:
        st.info("Waiting for sensor JSON samples from Phyphox...")

    result = run_live_analysis_tick()
    if result is not None:
        pred_class, pred_proba, health_score, estimated_rul_hours, trend, latest_row = result
        render_live_dashboard(pred_class, pred_proba, health_score, estimated_rul_hours, trend, latest_row)
    elif not st.session_state.live_buffer.empty:
        hist = analysis_history_df()
        if not hist.empty:
            last = hist.iloc[-1]
            fake_proba = np.array([last["p_good"], last["p_bad"], last["p_critical"]])
            render_live_dashboard(
                int(last["status"]), fake_proba, last["health"], last["rul_hours"],
                last["trend"], st.session_state.live_buffer.iloc[[-1]],
            )
        else:
            # Have sensor rows but no history yet — still force a prediction render
            latest_row = st.session_state.live_buffer.iloc[[-1]]
            medians = st.session_state.model_data.get("medians") if st.session_state.model_data else None
            X_prep = preprocess_features(st.session_state.live_buffer.iloc[-15:], medians=medians)
            if st.session_state.model_data:
                pc, pp = predict_health_status(st.session_state.model_data["model"], X_prep)
                feats = extract_features(st.session_state.live_buffer.iloc[-100:])
                hs, rul, tr = estimate_health_and_rul(feats, pp)
                render_live_dashboard(pc, pp, hs, rul, tr, latest_row)
    else:
        # Show raw JSON even when buffer empty so user can debug
        if st.session_state.latest_phyphox_json is not None:
            st.markdown('<div class="section-title">Raw Phyphox JSON (debug)</div>', unsafe_allow_html=True)
            st.json(st.session_state.latest_phyphox_json)
else:
    st.markdown("""
    <div class="live-banner">
        <div class="pulse-dot off"></div>
        IDLE — paste Phyphox HTTP URL above, Connect, then Start Live
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Uses real phone sensors only (no random demo data)**
    1. Open the **Predictive Maintenance** experiment in Phyphox
    2. Enable **Allow remote access** and copy the URL (e.g. `http://192.168.1.20:8080`)
    3. Paste it above → **Connect** → **Start Live**
    4. Cards and charts update from live `/get` JSON (hold still = Good, move = Bad, shake = Critical)
    """)

    if st.session_state.analysis_history:
        st.markdown('<div class="section-title">Last session visualizations</div>', unsafe_allow_html=True)
        hist = analysis_history_df()
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(hist.set_index("time_str")[["health"]].rename(columns={"health": "Health %"}), height=200)
        with c2:
            st.area_chart(
                hist.set_index("time_str")[["p_good", "p_bad", "p_critical"]].rename(
                    columns={"p_good": "Good", "p_bad": "Bad", "p_critical": "Critical"}
                ),
                height=200,
            )

st.markdown("---")
st.caption("Data source: Phyphox HTTP /get JSON · Model: XGBoost Good/Bad/Critical · Refresh: 1 Hz")

if is_streaming:
    time.sleep(1.0)
    st.rerun()
