import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="WaveCast | AI Wave Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom css injector
def inject_custom_css():
    st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

    /* Deep Ocean Theme styles */
    .stApp {
        background: linear-gradient(135deg, #050b14 0%, #0a1628 100%) !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Top Padding & Custom Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .glass-card:hover {
        border-color: rgba(0, 212, 255, 0.2) !important;
        box-shadow: 0 12px 40px 0 rgba(0, 212, 255, 0.08) !important;
        transform: translateY(-2px) !important;
    }

    /* Specific left border accents for card types */
    .accent-cyan {
        border-left: 4px solid #00d4ff !important;
    }
    .accent-teal {
        border-left: 4px solid #14b8a6 !important;
    }
    .accent-blue {
        border-left: 4px solid #0ea5e9 !important;
    }

    /* Text Gradient Cyan */
    .text-gradient-cyan {
        background: linear-gradient(90deg, #00d4ff 0%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        display: inline-block;
    }

    /* Custom buttons */
    div.stButton > button {
        border-radius: 9999px !important;
        padding: 12px 36px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Predict Button (Primary Glowing Cyan) */
    .predict-btn div.stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #0ea5e9 100%) !important;
        color: #050b14 !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3) !important;
    }

    .predict-btn div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.6) !important;
        color: #050b14 !important;
        border: none !important;
    }

    /* Secondary Buttons (Fill/Reset) */
    .sec-btn div.stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f1f5f9 !important;
    }

    .sec-btn div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(20, 184, 166, 0.4) !important;
        color: #14b8a6 !important;
        box-shadow: 0 0 15px rgba(20, 184, 166, 0.15) !important;
    }

    /* Status badge pill */
    .status-badge {
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.2);
        color: #00d4ff;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #00d4ff;
        border-radius: 50%;
        box-shadow: 0 0 10px #00d4ff;
        display: inline-block;
    }

    .status-dot.demo {
        background-color: #f59e0b;
        box-shadow: 0 0 10px #f59e0b;
    }

    .status-badge.demo {
        background: rgba(245, 158, 11, 0.08);
        border-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }

    /* Header UI */
    .nav-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 0;
        margin-bottom: 25px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .nav-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nav-logo-icon {
        font-size: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .nav-title {
        font-family: 'Outfit', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.1;
    }

    .nav-subtitle {
        font-size: 11px;
        color: rgba(255, 255, 255, 0.4);
        margin: 0;
    }

    /* Down-arrow Chevron bounce animation */
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-8px);
        }
        60% {
            transform: translateY(-4px);
        }
    }

    .chevron-down {
        animation: bounce 2s infinite;
        display: flex;
        justify-content: center;
        color: #00d4ff;
        font-size: 28px;
        margin: 20px 0;
    }

    /* Data Editor override background and glass feel */
    .stDataEditor {
        background-color: rgba(5, 11, 20, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* Wave Orb Styles */
    .orb-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 15px 0 25px 0;
    }

    .wave-orb {
        position: relative;
        width: 210px;
        height: 210px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.02);
        border: 3px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 0 35px rgba(0, 212, 255, 0.18), inset 0 0 20px rgba(0, 212, 255, 0.1);
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .wave-orb::after {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 50%;
        box-shadow: inset 0 0 25px rgba(0,0,0,0.7);
        pointer-events: none;
        z-index: 5;
    }

    .wave {
        position: absolute;
        bottom: 0;
        left: -50%;
        width: 200%;
        height: 200%;
        border-radius: 40%;
        z-index: 1;
    }

    .wave-1 {
        background: linear-gradient(180deg, rgba(0, 212, 255, 0.45) 0%, rgba(20, 184, 166, 0.65) 100%);
        transform: translateY(calc(100% - var(--fill-pct))) rotate(0deg);
        animation: rotate-wave 12s linear infinite;
    }

    .wave-2 {
        background: linear-gradient(180deg, rgba(14, 165, 233, 0.35) 0%, rgba(0, 212, 255, 0.55) 100%);
        transform: translateY(calc(96% - var(--fill-pct))) rotate(0deg);
        animation: rotate-wave 8s linear infinite reverse;
        opacity: 0.8;
    }

    .orb-content {
        position: relative;
        z-index: 10;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .orb-value {
        font-family: 'Outfit', sans-serif;
        font-size: 58px;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 2px 12px rgba(0,0,0,0.6);
        line-height: 0.95;
    }

    .orb-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 4px;
        text-shadow: 0 1px 4px rgba(0,0,0,0.6);
    }

    @keyframes rotate-wave {
        0% { transform: translateY(calc(100% - var(--fill-pct))) rotate(0deg); }
        100% { transform: translateY(calc(100% - var(--fill-pct))) rotate(360deg); }
    }

    /* KPI styling */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-top: 15px;
        width: 100%;
    }

    .kpi-box {
        background: rgba(255,255,255,0.015);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .kpi-box:hover {
        background: rgba(255,255,255,0.03);
        border-color: rgba(0,212,255,0.1);
    }

    .kpi-val {
        font-family: 'Outfit', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }

    .kpi-lbl {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255,255,255,0.5);
        margin-top: 2px;
    }

    /* Subtitle helper spacing */
    .section-desc {
        color: rgba(255, 255, 255, 0.6);
        font-size: 15px;
        margin-bottom: 24px;
        max-width: 600px;
    }
    
    .how-to-list {
        padding-left: 18px;
        margin: 0;
    }
    
    .how-to-list li {
        color: rgba(255, 255, 255, 0.7);
        font-size: 14px;
        margin-bottom: 10px;
        line-height: 1.5;
    }
    
    .how-to-list strong {
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Load resources (model & scalers)
@st.cache_resource
def load_prediction_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    
    # Check for possible model file locations
    model_path_keras = os.path.join(model_dir, "wave_cnn_lstm.keras")
    model_path_h5 = os.path.join(model_dir, "wave_prediction_model.h5")
    x_scaler_path = os.path.join(model_dir, "x_scaler.pkl")
    y_scaler_path = os.path.join(model_dir, "y_scaler.pkl")
    
    model = None
    feature_scaler = None
    target_scaler = None
    status = "Demo Mode"
    
    # Try importing Tensorflow safely
    try:
        from tensorflow.keras.models import load_model
        
        if os.path.exists(model_path_keras):
            model = load_model(model_path_keras, compile=False)
            status = "Model loaded (.keras)"
        elif os.path.exists(model_path_h5):
            model = load_model(model_path_h5, compile=False)
            status = "Model loaded (.h5)"
    except Exception as e:
        pass  # Tensorflow import or loading error, fallback to Demo Mode
        
    # Load scalers
    if os.path.exists(x_scaler_path):
        try:
            with open(x_scaler_path, "rb") as f:
                feature_scaler = pickle.load(f)
        except Exception:
            pass
            
    if os.path.exists(y_scaler_path):
        try:
            with open(y_scaler_path, "rb") as f:
                target_scaler = pickle.load(f)
        except Exception:
            pass
            
    if model is None or feature_scaler is None or target_scaler is None:
        return None, None, None, "Demo Mode"
    return model, feature_scaler, target_scaler, status

# Load model/scalers
model, feature_scaler, target_scaler, model_status = load_prediction_resources()

# Initialize styling
inject_custom_css()

# Default generator for Reset
def generate_reset_matrix():
    data = {
        "Wave (m)": [0.0] * 10,
        "Wind (m/s)": [0.0] * 10,
        "Pressure (hPa)": [1013.0] * 10,
        "Temp (°C)": [20.0] * 10
    }
    indices = [f"t-{9-i}" for i in range(10)]
    df = pd.DataFrame(data, index=indices, dtype=float)
    return df

# Synthetic sample generator
def generate_sample_matrix():
    np.random.seed(int(pd.Timestamp.now().timestamp()) % 100000)
    t = np.arange(10)
    
    # Wave height: smooth sine wave + minor noise (1.0m to 2.5m)
    wave = 1.6 + 0.5 * np.sin(t * 0.4) + np.random.normal(0, 0.08, size=10)
    
    # Wind speed: correlates heavily with wave height (e.g. 10 + 4 * wave)
    wind = 8.0 + 3.5 * np.sin(t * 0.4 - 0.2) + np.random.normal(0, 0.7, size=10)
    
    # Pressure: anti-correlated with wave swells
    pressure = 1014.0 - 6.0 * np.sin(t * 0.4 - 0.1) + np.random.normal(0, 0.4, size=10)
    
    # Temperature: seasonal stable ocean temperature
    temp = 19.5 + 1.2 * np.cos(t * 0.3) + np.random.normal(0, 0.15, size=10)
    
    # Clamping for realistic values
    wave = np.clip(wave, 0.1, 5.0)
    wind = np.clip(wind, 0.5, 25.0)
    pressure = np.clip(pressure, 985.0, 1025.0)
    temp = np.clip(temp, 12.0, 32.0)
    
    data = {
        "Wave (m)": np.round(wave, 2),
        "Wind (m/s)": np.round(wind, 2),
        "Pressure (hPa)": np.round(pressure, 2),
        "Temp (°C)": np.round(temp, 2)
    }
    
    indices = [f"t-{9-i}" for i in range(10)]
    df = pd.DataFrame(data, index=indices, dtype=float)
    return df

# Deterministic demo predictor
def run_demo_prediction(matrix_data):
    # wave is col 0, wind is col 1, pressure is col 2, temp is col 3
    mean_wave = np.mean(matrix_data[:, 0])
    latest_wave = matrix_data[-1, 0]
    max_wind = np.max(matrix_data[:, 1])
    min_pressure = np.min(matrix_data[:, 2])
    
    # Wind speed contribution (higher wind = higher wave)
    wind_factor = 0.045 * max_wind
    
    # Atmospheric pressure contribution (dropping pressure = storm swell)
    pressure_factor = max(0.0, (1013.0 - min_pressure) * 0.025)
    
    # Combine signals
    pred = 0.65 * latest_wave + 0.35 * mean_wave + wind_factor + pressure_factor
    
    # Small cyclical variation
    pred += 0.06 * np.sin(latest_wave * 1.5)
    
    # Bound to a 0-6 meter range
    return float(np.clip(pred, 0.1, 6.0))

# Standard prediction processor
def predict_wave_height(matrix_df):
    data = matrix_df.to_numpy(dtype=float)
    
    # Validation check for NaNs or infinites
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data editor matrix contains invalid numeric cells or empty inputs.")
        
    if model is not None and feature_scaler is not None and target_scaler is not None:
        try:
            # Scale features using x_scaler.pkl
            scaled_data = feature_scaler.transform(data)
            # Reshape for CNN-LSTM input format: (1, 10, 4)
            reshaped_data = scaled_data.reshape(1, 10, 4)
            # Run Tensorflow model inference
            prediction_scaled = model.predict(reshaped_data, verbose=0)
            # Inverse scale predicted output using y_scaler.pkl
            prediction = target_scaler.inverse_transform(prediction_scaled)
            return float(np.clip(prediction[0][0], 0.1, 6.0))
        except Exception as e:
            # If tensor operations fail, gracefully run demo fallback
            return run_demo_prediction(data)
    else:
        # Load fallback demo predictor
        return run_demo_prediction(data)

# Initialize Session State
if "matrix" not in st.session_state:
    st.session_state.matrix = generate_sample_matrix()
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "history" not in st.session_state:
    st.session_state.history = []

# ==================== LAYOUT: HEADER / NAV ====================
st.markdown(f"""
<div class="nav-header">
    <div class="nav-logo">
        <div class="nav-logo-icon">🌊</div>
        <div>
            <div class="nav-title">WaveCast</div>
            <div class="nav-subtitle">CNN-LSTM Ocean Forecast</div>
        </div>
    </div>
    <div class="status-badge { 'demo' if 'Demo' in model_status else '' }">
        <span class="status-dot { 'demo' if 'Demo' in model_status else '' }"></span>
        {model_status}
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== LAYOUT: HERO SECTION ====================
st.markdown("""
<div style="text-align: center; margin-top: 15px; margin-bottom: 25px;">
    <h1 class="text-gradient-cyan" style="font-size: 52px; margin-bottom: 12px; font-weight: 700; tracking-tight;">
        Wave Height Prediction
    </h1>
    <p style="color: rgba(255, 255, 255, 0.65); font-size: 18px; max-width: 650px; margin: 0 auto 20px auto; line-height: 1.5; font-weight: 300;">
        Advanced deep learning system utilizing CNN-LSTM networks to forecast oceanic wave heights and surface conditions with precision.
    </p>
    <div class="chevron-down">▼</div>
</div>
""", unsafe_allow_html=True)

# Main Grid (2 Columns: Left is input data table, Right is result visualization & chart)
col_left, col_right = st.columns([11, 10], gap="large")

with col_left:
    # Glassmorphic card wrap for input analyzer
    st.markdown('<div class="glass-card accent-cyan">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0; margin-bottom: 8px;'>Analyze Ocean Data</h3>", unsafe_allow_html=True)
    st.markdown("<p class='section-desc'>Modify the 10-step sliding history window below. Double-click cells to edit or adjust feature sliders.</p>", unsafe_allow_html=True)
    
    # Data editor matrix
    edited_df = st.data_editor(
        st.session_state.matrix,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Wave (m)": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=10.0),
            "Wind (m/s)": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=50.0),
            "Pressure (hPa)": st.column_config.NumberColumn(format="%.2f", min_value=900.0, max_value=1100.0),
            "Temp (°C)": st.column_config.NumberColumn(format="%.2f", min_value=-10.0, max_value=45.0),
        }
    )
    
    # Store updated state to reflect changes
    st.session_state.matrix = edited_df

    # Button Row Layout
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2], gap="small")
    
    with btn_col1:
        st.markdown('<div class="sec-btn">', unsafe_allow_html=True)
        if st.button("Fill Sample"):
            st.session_state.matrix = generate_sample_matrix()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with btn_col2:
        st.markdown('<div class="sec-btn">', unsafe_allow_html=True)
        if st.button("Reset"):
            st.session_state.matrix = generate_reset_matrix()
            st.session_state.prediction = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with btn_col3:
        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
        predict_click = st.button("Predict Wave Height")
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_click:
        try:
            with st.spinner("Running CNN-LSTM forecast..."):
                pred = predict_wave_height(st.session_state.matrix)
                st.session_state.prediction = pred
                
                # Append to history
                timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
                latest_wave = st.session_state.matrix.iloc[-1]["Wave (m)"]
                st.session_state.history.append({
                    "time": timestamp,
                    "latest_in": f"{latest_wave:.2f} m",
                    "forecast": f"{pred:.2f} m"
                })
                
                # Prune history to last 5 runs
                if len(st.session_state.history) > 5:
                    st.session_state.history.pop(0)
                    
                st.rerun()
        except Exception as e:
            st.error(f"Prediction Failed: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Glassmorphic card wrap for prediction visual result
    st.markdown('<div class="glass-card accent-teal">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0; margin-bottom: 5px;'>Inference Outcome</h3>", unsafe_allow_html=True)
    
    pred_val = st.session_state.prediction
    fill_pct = min(100.0, max(0.0, (pred_val / 6.0) * 100.0)) if pred_val is not None else 15.0
    orb_text = f"{pred_val:.2f}" if pred_val is not None else "—"
    
    # Visual columns for Orb + Mini KPI statistics
    orb_col, stats_col = st.columns([8, 7])
    
    with orb_col:
        # Inject styling with dynamic percentage and render custom animated orb
        st.markdown(f"""
        <div class="orb-container">
            <div class="wave-orb" style="--fill-pct: {fill_pct}%;">
                <div class="wave wave-1"></div>
                <div class="wave wave-2"></div>
                <div class="orb-content">
                    <div class="orb-value">{orb_text}</div>
                    <div class="orb-label">Meters</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with stats_col:
        st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
        t0_wave = st.session_state.matrix.iloc[-1]["Wave (m)"]
        t0_wind = st.session_state.matrix.iloc[-1]["Wind (m/s)"]
        t0_press = st.session_state.matrix.iloc[-1]["Pressure (hPa)"]
        
        # Display custom metrics
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-val">{t0_wave:.2f} m</div>
            <div class="kpi-lbl">Latest Wave (t-0)</div>
        </div>
        <div class="kpi-box" style="margin-top: 10px;">
            <div class="kpi-val">{t0_wind:.1f} m/s</div>
            <div class="kpi-lbl">Latest Wind (t-0)</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Render Plotly Forecast Chart
    waves_history = st.session_state.matrix["Wave (m)"].tolist()
    timesteps = [f"t-{9-i}" for i in range(10)]
    x_labels = timesteps + ["Forecast (t+1)"]
    
    fig = go.Figure()
    # History Trace (Teal)
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=waves_history,
        mode="lines+markers",
        name="Wave History",
        line=dict(color="#14b8a6", width=3),
        marker=dict(size=8, color="#0d9488"),
        hovertemplate="Wave Height: %{y:.2f} m<extra></extra>"
    ))
    
    # Overlay point trace (Cyan)
    if pred_val is not None:
        fig.add_trace(go.Scatter(
            x=[timesteps[-1], "Forecast (t+1)"],
            y=[waves_history[-1], pred_val],
            mode="lines",
            showlegend=False,
            line=dict(color="#00d4ff", width=2, dash="dash"),
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=["Forecast (t+1)"],
            y=[pred_val],
            mode="markers+text",
            name="Forecast Value",
            marker=dict(size=13, color="#00d4ff", line=dict(color="#ffffff", width=2)),
            text=[f"{pred_val:.2f}m"],
            textposition="top center",
            hovertemplate="Forecast: %{y:.2f} m<extra></extra>"
        ))
        
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="rgba(255,255,255,0.5)", size=10),
            showgrid=True,
            showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="rgba(255,255,255,0.5)", size=10),
            showgrid=True,
            showline=False,
        ),
        legend=dict(
            font=dict(color="#ffffff", size=10),
            bgcolor="rgba(5, 11, 20, 0.4)",
            bordercolor="rgba(255,255,255,0.05)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=10, b=10),
        height=180,
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== LAYOUT: HISTORICAL LOGS SIDEBAR / COLLAPSED ====================
if st.session_state.history:
    with st.expander("📊 Inference History Log", expanded=False):
        st.markdown('<div class="glass-card accent-blue" style="padding: 15px !important;">', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== LAYOUT: FEATURE INFO CARDS ====================
st.markdown("<h3 style='margin-top: 20px; margin-bottom: 15px;'>Ocean Feature Parameters</h3>", unsafe_allow_html=True)
card_col1, card_col2, card_col3 = st.columns(3, gap="medium")

with card_col1:
    st.markdown("""
    <div class="glass-card accent-cyan" style="height: 100%;">
        <h4 style="margin-top: 0; margin-bottom: 8px; color: #00d4ff;">🌊 Wave Height</h4>
        <p style="font-size: 13px; color: rgba(255, 255, 255, 0.65); line-height: 1.5; margin: 0;">
            The previous recorded wave height. This provides the primary temporal context needed by the CNN-LSTM neural network to forecast the next sequence.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
with card_col2:
    st.markdown("""
    <div class="glass-card accent-teal" style="height: 100%;">
        <h4 style="margin-top: 0; margin-bottom: 8px; color: #14b8a6;">💨 Wind Speed</h4>
        <p style="font-size: 13px; color: rgba(255, 255, 255, 0.65); line-height: 1.5; margin: 0;">
            The speed of surface wind. Wind transfers kinetic energy to the water column, and sustained high winds are the principal driver of growing wave swells.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
with card_col3:
    st.markdown("""
    <div class="glass-card accent-blue" style="height: 100%;">
        <h4 style="margin-top: 0; margin-bottom: 8px; color: #0ea5e9;">📉 Air Pressure</h4>
        <p style="font-size: 13px; color: rgba(255, 255, 255, 0.65); line-height: 1.5; margin: 0;">
            Atmospheric air pressure. Rapid storm developments are marked by falling pressure drops, initiating large weather swells and intense wave conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== LAYOUT: HOW TO USE ====================
st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card accent-blue">
    <h3 style="margin-top: 0; margin-bottom: 12px;">How to Use WaveCast</h3>
    <ul class="how-to-list">
        <li><strong>1. Provide Time Series:</strong> Input a sliding window of exactly 10 consecutive historical timesteps in the data table.</li>
        <li><strong>2. Set Variables:</strong> Each timestep requires wave height (m), wind speed (m/s), atmospheric pressure (hPa), and air temperature (°C). You can use "Fill Sample" to populate realistic mock values instantly.</li>
        <li><strong>3. Execute CNN-LSTM Predictor:</strong> Click the glowing "Predict Wave Height" button. The system processes the input window and computes the wave height forecast for the next hour (t+1).</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ==================== LAYOUT: FOOTER ====================
st.markdown("""
<div style="border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 20px; margin-top: 30px; text-align: center; font-size: 12px; color: rgba(255, 255, 255, 0.4);">
    © 2026 Wave Prediction System | Powered by Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)
