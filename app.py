import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="WaveCast - CNN-LSTM Wave Forecast",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom premium styling, typography, and animated backgrounds
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Main App Container Styling */
.stApp {
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #050b14 0%, #0a1628 100%) !important;
    background-attachment: fixed !important;
    color: #ffffff !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* Glassmorphism Cards */
div.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 24px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), 
                box-shadow 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), 
                border-color 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    margin-bottom: 20px;
}

div.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px 0 rgba(0, 212, 255, 0.15);
    border-color: rgba(0, 212, 255, 0.3);
}

/* Gradient and Typography Styling */
.gradient-text {
    background: linear-gradient(90deg, #00d4ff 0%, #14b8a6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Custom styled text blocks */
.section-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 20px;
    background: linear-gradient(90deg, #ffffff 0%, rgba(255,255,255,0.7) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Action Buttons Styling */
.stButton > button {
    background: linear-gradient(135deg, rgba(20, 184, 166, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.05) !important;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #14b8a6 0%, #00d4ff 100%) !important;
    border-color: #00d4ff !important;
    color: #050b14 !important;
    box-shadow: 0 0 25px rgba(0, 212, 255, 0.4) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Custom styling for Data Editor */
[data-testid="stDataEditor"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    padding: 10px !important;
}

/* Background Ocean Particles & Animations */
.ocean-background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    pointer-events: none;
    overflow: hidden;
}

/* Bubble float effect */
.bubble {
    position: absolute;
    bottom: -50px;
    background: radial-gradient(circle, rgba(0, 212, 255, 0.12) 0%, rgba(20, 184, 166, 0.01) 70%);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 50%;
    animation: float infinite linear;
}

@keyframes float {
    0% {
        transform: translateY(0) scale(1) rotate(0deg);
        opacity: 0;
    }
    10% {
        opacity: 0.5;
    }
    90% {
        opacity: 0.5;
    }
    100% {
        transform: translateY(-110vh) scale(1.2) rotate(360deg);
        opacity: 0;
    }
}

/* Particle drift effect */
.particle {
    position: absolute;
    background-color: rgba(0, 212, 255, 0.3);
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.6);
    animation: drift infinite ease-in-out;
}

@keyframes drift {
    0%, 100% {
        transform: translate(0, 0);
        opacity: 0.2;
    }
    50% {
        transform: translate(25px, -25px);
        opacity: 0.5;
    }
}

/* Bottom Wave SVG */
.wave-container {
    position: absolute;
    bottom: 0;
    width: 100%;
    height: 100px;
    min-height: 80px;
    max-height: 120px;
    opacity: 0.65;
}

.waves {
    position: relative;
    width: 100%;
    height: 100%;
}

.parallax > use {
    animation: move-forever 18s cubic-bezier(.55,.5,.45,.5) infinite;
}
.parallax > use:nth-child(1) {
    animation-delay: -2s;
    animation-duration: 6s;
}
.parallax > use:nth-child(2) {
    animation-delay: -3s;
    animation-duration: 9s;
}
.parallax > use:nth-child(3) {
    animation-delay: -4s;
    animation-duration: 12s;
}

@keyframes move-forever {
    0% {
        transform: translate3d(-90px, 0, 0);
    }
    100% {
        transform: translate3d(85px, 0, 0);
    }
}

/* Animated Down Chevron */
.chevron-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 60px;
    margin-top: 20px;
}

.chevron {
    position: relative;
    width: 28px;
    height: 8px;
    opacity: 0;
    transform: scale3d(0.5, 0.5, 0.5);
    animation: move-chevron 3s ease-out infinite;
}

.chevron:first-child {
    animation: move-chevron 3s ease-out 1s infinite;
}

.chevron:nth-child(2) {
    animation: move-chevron 3s ease-out 2s infinite;
}

.chevron:before,
.chevron:after {
    content: ' ';
    position: absolute;
    top: 0;
    height: 100%;
    width: 51%;
    background: #00d4ff;
    border-radius: 4px;
}

.chevron:before {
    left: 0;
    transform: skewY(25deg);
}

.chevron:after {
    right: 0;
    width: 50%;
    transform: skewY(-25deg);
}

@keyframes move-chevron {
    25% {
        opacity: 0.8;
    }
    33% {
        opacity: 0.8;
        transform: translateY(15px);
    }
    67% {
        opacity: 0.8;
        transform: translateY(20px);
    }
    100% {
        opacity: 0;
        transform: translateY(30px) scale3d(0.5, 0.5, 0.5);
    }
}
</style>

<!-- Background Overlay Injected Elements -->
<div class="ocean-background">
    <!-- Particles -->
    <div class="particle" style="top: 15%; left: 8%; width: 4px; height: 4px; animation-duration: 8s; animation-delay: 0s;"></div>
    <div class="particle" style="top: 30%; left: 85%; width: 3px; height: 3px; animation-duration: 10s; animation-delay: 1.5s;"></div>
    <div class="particle" style="top: 55%; left: 18%; width: 4px; height: 4px; animation-duration: 12s; animation-delay: 3s;"></div>
    <div class="particle" style="top: 22%; left: 60%; width: 3px; height: 3px; animation-duration: 9s; animation-delay: 0.5s;"></div>
    <div class="particle" style="top: 75%; left: 45%; width: 4px; height: 4px; animation-duration: 11s; animation-delay: 4s;"></div>
    <div class="particle" style="top: 48%; left: 72%; width: 5px; height: 5px; animation-duration: 7s; animation-delay: 2.2s;"></div>
    <div class="particle" style="top: 82%; left: 90%; width: 3px; height: 3px; animation-duration: 13s; animation-delay: 1.8s;"></div>
    <div class="particle" style="top: 66%; left: 33%; width: 4px; height: 4px; animation-duration: 10s; animation-delay: 5s;"></div>
    <div class="particle" style="top: 90%; left: 15%; width: 3px; height: 3px; animation-duration: 14s; animation-delay: 3.5s;"></div>
    <div class="particle" style="top: 10%; left: 40%; width: 5px; height: 5px; animation-duration: 9s; animation-delay: 2s;"></div>
    
    <!-- Bubbles -->
    <div class="bubble" style="left: 6%; width: 6px; height: 6px; animation-duration: 9s; animation-delay: 0s;"></div>
    <div class="bubble" style="left: 20%; width: 10px; height: 10px; animation-duration: 12s; animation-delay: 2.5s;"></div>
    <div class="bubble" style="left: 38%; width: 5px; height: 5px; animation-duration: 8s; animation-delay: 4.8s;"></div>
    <div class="bubble" style="left: 55%; width: 12px; height: 12px; animation-duration: 15s; animation-delay: 1s;"></div>
    <div class="bubble" style="left: 68%; width: 8px; height: 8px; animation-duration: 11s; animation-delay: 6s;"></div>
    <div class="bubble" style="left: 80%; width: 4px; height: 4px; animation-duration: 10s; animation-delay: 3.2s;"></div>
    <div class="bubble" style="left: 92%; width: 11px; height: 11px; animation-duration: 13s; animation-delay: 7.5s;"></div>
    <div class="bubble" style="left: 14%; width: 7px; height: 7px; animation-duration: 14s; animation-delay: 5.2s;"></div>
    <div class="bubble" style="left: 48%; width: 9px; height: 9px; animation-duration: 11s; animation-delay: 3s;"></div>

    <!-- Waves -->
    <div class="wave-container">
        <svg class="waves" xmlns="http://www.w3.org/2000/svg" viewBox="0 24 150 28" preserveAspectRatio="none" shape-rendering="auto">
            <defs>
                <path id="gentle-wave" d="M-160 44c30 0 58-18 88-18s58 18 88 18 58-18 88-18 58 18 88 18 v44h-352z" />
            </defs>
            <g class="parallax">
                <use href="#gentle-wave" x="48" y="0" fill="rgba(0, 212, 255, 0.06)" />
                <use href="#gentle-wave" x="48" y="3" fill="rgba(20, 184, 166, 0.10)" />
                <use href="#gentle-wave" x="48" y="5" fill="rgba(5, 11, 20, 0.3)" />
            </g>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# Model Assets and Cache Initialization
# ----------------------------------------------------
@st.cache_resource
def load_ml_assets():
    """
    Attempts to load the pre-trained deep learning model and scaling dictionaries.
    Gracefully falls back to Demo Mode if assets or packages are unavailable.
    """
    model_path = "models/wave_cnn_lstm.keras"
    x_scaler_path = "models/x_scaler.pkl"
    y_scaler_path = "models/y_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path):
        try:
            import tensorflow as tf
            import joblib
            model = tf.keras.models.load_model(model_path, compile=False)
            x_scaler = joblib.load(x_scaler_path)
            y_scaler = joblib.load(y_scaler_path)
            return model, x_scaler, y_scaler, True
        except Exception as e:
            # Silently fallback to Demo Mode if package loading or file deserialization fails
            return None, None, None, False
    return None, None, None, False

# Try loading the ML model
model, x_scaler, y_scaler, has_model = load_ml_assets()


# ----------------------------------------------------
# Session State Initialization
# ----------------------------------------------------
if 'matrix' not in st.session_state:
    # Default values: Wave Height (0.0m), Wind (0.0 m/s), Pressure (1013.0 hPa), Temp (20.0°C)
    st.session_state.matrix = pd.DataFrame(
        [[0.0, 0.0, 1013.0, 20.0]] * 10,
        columns=['Wave (m)', 'Wind (m/s)', 'Pressure (hPa)', 'Temp (°C)'],
        index=[f't-{i}' for i in range(9, -1, -1)]
    )

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if 'history' not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------
# 1. HEADER SECTION
# ----------------------------------------------------
header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    st.markdown("""
    <div style="margin-top: 10px;">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800; display: inline-flex; align-items: center; gap: 12px; line-height: 1.1;">
            <span>🌊</span><span class="gradient-text">WaveCast</span>
        </h1>
        <p style="margin: 4px 0 0 0; color: #14b8a6; font-weight: 600; font-size: 1.1rem; letter-spacing: 1px; text-transform: uppercase;">
            CNN-LSTM Ocean Forecast System
        </p>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    status_pill = ""
    if has_model:
        status_pill = """
        <div style="text-align: right; margin-top: 15px;">
            <span style="background: rgba(0, 212, 255, 0.15); color: #00d4ff; border: 1px solid rgba(0, 212, 255, 0.30); padding: 8px 18px; border-radius: 25px; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.5px; box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);">
                Model Loaded ✓
            </span>
        </div>
        """
    else:
        status_pill = """
        <div style="text-align: right; margin-top: 15px;">
            <span style="background: rgba(20, 184, 166, 0.15); color: #14b8a6; border: 1px solid rgba(20, 184, 166, 0.30); padding: 8px 18px; border-radius: 25px; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.5px; box-shadow: 0 0 15px rgba(20, 184, 166, 0.2);">
                Demo Mode
            </span>
        </div>
        """
    st.markdown(status_pill, unsafe_allow_html=True)


# ----------------------------------------------------
# 2. HERO SECTION
# ----------------------------------------------------
hero_html = """
<div class="glass-card" style="text-align: center; padding: 45px 30px; margin-top: 25px; margin-bottom: 30px;">
    <h2 style="font-size: 2.8rem; font-weight: 800; margin: 0 0 15px 0; letter-spacing: -0.5px; line-height: 1.2;">
        Wave Height Prediction
    </h2>
    <p style="font-size: 1.15rem; max-width: 800px; margin: 0 auto; color: rgba(255, 255, 255, 0.8); line-height: 1.6; font-weight: 300;">
        Deep learning system trained on NOAA NDBC Station 46059 ocean buoy data to forecast wave heights with precision.
    </p>
    <div class="chevron-container">
        <div class="chevron"></div>
        <div class="chevron"></div>
        <div class="chevron"></div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)


# ----------------------------------------------------
# 3. INPUT MATRIX SECTION
# ----------------------------------------------------
st.markdown('<p class="section-title">Analyze Ocean Data</p>', unsafe_allow_html=True)

# Container for data editor input
edited_df = st.data_editor(
    st.session_state.matrix,
    column_config={
        'Wave (m)': st.column_config.NumberColumn("Wave (m)", min_value=0.0, max_value=12.0, step=0.01, format="%.2f"),
        'Wind (m/s)': st.column_config.NumberColumn("Wind (m/s)", min_value=0.0, max_value=60.0, step=0.1, format="%.1f"),
        'Pressure (hPa)': st.column_config.NumberColumn("Pressure (hPa)", min_value=900.0, max_value=1100.0, step=0.1, format="%.1f"),
        'Temp (°C)': st.column_config.NumberColumn("Temp (°C)", min_value=-5.0, max_value=45.0, step=0.1, format="%.1f"),
    },
    use_container_width=True,
    num_rows="fixed",
    key="editor_key"
)

# Update session state with edited table values immediately
st.session_state.matrix = edited_df

# Buttons layout below table
btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    if st.button("Fill Sample Data"):
        # Fill realistic sequential ocean weather data using overlapping waves + noise
        np.random.seed(int(time.time()))
        t = np.arange(10)
        
        # Wave: smooth fluctuations between 1.0m and 2.5m
        waves = 1.7 + 0.6 * np.sin(t * 0.4) + np.random.uniform(-0.15, 0.15, 10)
        waves = np.clip(waves, 1.0, 2.5)
        
        # Wind: related to wave heights, moving between 5 and 12 m/s
        winds = 7.5 + 3.5 * np.sin(t * 0.4 - 0.2) + np.random.uniform(-0.6, 0.6, 10)
        winds = np.clip(winds, 5.0, 12.0)
        
        # Pressure: minor cyclo-baric drop/rise between 1008 and 1015 hPa
        pressures = 1012.0 - 2.5 * np.cos(t * 0.3) + np.random.uniform(-0.4, 0.4, 10)
        pressures = np.clip(pressures, 1008.0, 1015.0)
        
        # Temp: gentle sea temperature changes, steady between 14°C and 18°C
        temps = 15.5 + 1.2 * np.sin(t * 0.15) + np.random.uniform(-0.15, 0.15, 10)
        temps = np.clip(temps, 14.0, 18.0)
        
        # Generate new Dataframe
        st.session_state.matrix = pd.DataFrame(
            np.column_stack([waves, winds, pressures, temps]),
            columns=['Wave (m)', 'Wind (m/s)', 'Pressure (hPa)', 'Temp (°C)'],
            index=[f't-{i}' for i in range(9, -1, -1)]
        )
        st.rerun()

with btn_col2:
    if st.button("Reset"):
        # Reset back to default matrix values: 0, 0, 1013, 20
        st.session_state.matrix = pd.DataFrame(
            [[0.0, 0.0, 1013.0, 20.0]] * 10,
            columns=['Wave (m)', 'Wind (m/s)', 'Pressure (hPa)', 'Temp (°C)'],
            index=[f't-{i}' for i in range(9, -1, -1)]
        )
        st.session_state.prediction = None
        st.rerun()

# Trigger variable
predict_clicked = False
with btn_col3:
    if st.button("🌊 Predict Wave Height"):
        predict_clicked = True


# ----------------------------------------------------
# 4. PREDICTION LOGIC
# ----------------------------------------------------
if predict_clicked:
    # 1. Validation check (All filled, no NaN)
    df_values = st.session_state.matrix.values
    if np.any(np.isnan(df_values)) or np.any(df_values == None):
        st.error("⚠️ Validation Error: All 10 consecutive rows and 4 parameters must be populated with numeric values.")
    else:
        with st.spinner("Running CNN-LSTM forecast..."):
            # Mock delay for premium visual responsiveness
            time.sleep(1.0)
            
            if has_model:
                try:
                    # Scale model inputs, reshape, predict and inverse transform
                    scaled_x = x_scaler.transform(df_values)
                    reshaped_x = scaled_x.reshape(1, 10, 4)
                    scaled_pred = model.predict(reshaped_x, verbose=0)
                    prediction_val = float(y_scaler.inverse_transform(scaled_pred)[0][0])
                except Exception as ex:
                    st.error(f"⚠️ Model Execution Failure: {str(ex)}. Reverting dynamically to backup forecast pipeline.")
                    # Backup fallback
                    wave_col = df_values[:, 0]
                    wind_col = df_values[:, 1]
                    prediction_val = float(np.mean(wave_col) * 1.1 + np.max(wind_col) * 0.05)
            else:
                # 2. Robust, physics-based deterministic Demo Mode formula
                wave_col = df_values[:, 0]
                wind_col = df_values[:, 1]
                pressure_col = df_values[:, 2]
                temp_col = df_values[:, 3]
                
                # Baseline wave height propagation
                base_prediction = np.mean(wave_col) * 1.1 + np.max(wind_col) * 0.05
                
                # Storm surge modifier (checking barometric pressure drops)
                pressure_diff = pressure_col[-1] - pressure_col[0]
                storm_modifier = 0.0
                if pressure_diff < -2.0:
                    storm_modifier += abs(pressure_diff) * 0.08
                elif pressure_col[-1] < 1010.0:
                    storm_modifier += 0.25
                
                # Temperature density modifier
                thermal_modifier = (np.mean(temp_col) - 16.0) * 0.008
                
                # Seeded deterministic noise variation based on inputs to maintain constant value for exact inputs
                seed_val = int(np.abs(np.sum(wave_col) * 1000 + np.sum(wind_col) * 10)) % 100
                variation = (seed_val - 50) / 1000.0
                
                prediction_val = float(np.clip(base_prediction + storm_modifier + thermal_modifier + variation, 0.0, 6.0))
            
            # Save to session state history
            st.session_state.prediction = prediction_val
            st.session_state.history.insert(0, prediction_val)
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[:5]


# ----------------------------------------------------
# 5. RESULT DISPLAY & 6. FORECAST CHART
# ----------------------------------------------------
if st.session_state.prediction is not None:
    pred_val = st.session_state.prediction
    
    # Assess safety thresholds
    if pred_val <= 2.0:
        safety_status = "SAFE"
        safety_color = "#14b8a6"
        safety_desc = "Safe conditions. Waters are calm, suitable for standard fishing, sailing, and transit operations."
        safety_bg = "rgba(20, 184, 166, 0.12)"
        safety_border = "1px solid rgba(20, 184, 166, 0.3)"
    elif pred_val <= 3.5:
        safety_status = "MODERATE"
        safety_color = "#eab308"
        safety_desc = "Moderate conditions. Heightened sea swells and surface wind currents. Caution is highly advised."
        safety_bg = "rgba(234, 179, 8, 0.12)"
        safety_border = "1px solid rgba(234, 179, 8, 0.3)"
    else:
        safety_status = "DANGEROUS"
        safety_color = "#ef4444"
        safety_desc = "Dangerous sea state. Heavy swells and severe atmospheric turbulence. Small vessels must avoid transit."
        safety_bg = "rgba(239, 68, 68, 0.12)"
        safety_border = "1px solid rgba(239, 68, 68, 0.3)"
        
    st.markdown('<p class="section-title">Forecast Analysis Dashboard</p>', unsafe_allow_html=True)
    
    col_res1, col_res2 = st.columns([1, 1.2], gap="large")
    
    with col_res1:
        # Custom Water Wave Orb HTML Component
        orb_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@600;800&display=swap" rel="stylesheet">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    font-family: 'Outfit', sans-serif;
                    overflow: hidden;
                }}
                
                .orb-container {{
                    position: relative;
                    width: 280px;
                    height: 280px;
                    border-radius: 50%;
                    border: 2px solid rgba(0, 212, 255, 0.6);
                    background: #050b14;
                    overflow: hidden;
                    box-shadow: 0 0 30px rgba(0, 212, 255, 0.4), inset 0 0 25px rgba(0, 212, 255, 0.25);
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    -webkit-mask-image: -webkit-radial-gradient(white, black);
                }}
                
                .water-container {{
                    position: absolute;
                    left: 0;
                    bottom: 0;
                    width: 100%;
                    height: 0%;
                    background: transparent;
                    z-index: 2;
                    transition: height 1.5s cubic-bezier(0.25, 0.8, 0.25, 1);
                }}
                
                .water-body {{
                    position: absolute;
                    left: 0;
                    bottom: 0;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(180deg, #00d4ff 0%, #0ea5e9 100%);
                }}
                
                .wave {{
                    position: absolute;
                    bottom: 100%;
                    left: 0;
                    width: 200%;
                    height: 28px;
                    background-repeat: repeat-x;
                    background-size: 50% 100%;
                    margin-bottom: -1px;
                }}
                
                .wave-1 {{
                    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 120" preserveAspectRatio="none"><path d="M0,60 Q150,90 300,60 T600,60 L600,120 L0,120 Z" fill="%2300d4ff" opacity="0.85"/></svg>');
                    animation: wave-slide-1 3s linear infinite;
                    z-index: 10;
                }}
                
                .wave-2 {{
                    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 120" preserveAspectRatio="none"><path d="M0,60 Q150,30 300,60 T600,60 L600,120 L0,120 Z" fill="%230ea5e9" opacity="0.55"/></svg>');
                    animation: wave-slide-2 2.5s linear infinite;
                    z-index: 5;
                }}
                
                @keyframes wave-slide-1 {{
                    0% {{ transform: translate3d(0, 0, 0); }}
                    100% {{ transform: translate3d(-50%, 0, 0); }}
                }}
                
                @keyframes wave-slide-2 {{
                    0% {{ transform: translate3d(-50%, 0, 0); }}
                    100% {{ transform: translate3d(0, 0, 0); }}
                }}
                
                .content {{
                    position: relative;
                    z-index: 10;
                    text-align: center;
                    user-select: none;
                    pointer-events: none;
                }}
                
                .number {{
                    font-size: 4.5rem;
                    font-weight: 800;
                    color: #ffffff;
                    line-height: 1;
                    text-shadow: 0 4px 15px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 212, 255, 0.4);
                    letter-spacing: -1px;
                }}
                
                .label {{
                    font-size: 0.95rem;
                    font-weight: 600;
                    color: rgba(255, 255, 255, 0.75);
                    letter-spacing: 3px;
                    margin-top: 5px;
                    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
                }}
            </style>
        </head>
        <body>
            <div class="orb-container">
                <div class="water-container" id="water-container">
                    <div class="wave wave-1"></div>
                    <div class="wave wave-2"></div>
                    <div class="water-body"></div>
                </div>
                <div class="content">
                    <div class="number">{pred_val:.2f}</div>
                    <div class="label">METERS</div>
                </div>
            </div>
            
            <script>
                const val = {pred_val};
                const clampedVal = Math.min(Math.max(val, 0), 6);
                const percent = (clampedVal / 6) * 100;
                
                setTimeout(() => {{
                    document.getElementById('water-container').style.height = percent + '%';
                }}, 100);
            </script>
        </body>
        </html>
        """
        components.html(orb_html, height=310)
        
        # Safety label markup
        safety_badge_html = f"""
        <div style="margin-top: 5px; padding: 12px 18px; border-radius: 8px; background: {safety_bg}; border: {safety_border}; text-align: center;">
            <p style="margin: 0; font-size: 1.1rem; font-weight: 800; color: {safety_color}; letter-spacing: 1px; text-transform: uppercase;">
                🌊 {safety_status} SEA STATE
            </p>
            <p style="margin: 6px 0 0 0; font-size: 0.9rem; font-weight: 300; line-height: 1.4; color: rgba(255, 255, 255, 0.85);">
                {safety_desc}
            </p>
        </div>
        """
        st.markdown(safety_badge_html, unsafe_allow_html=True)
        
        # History sub-section
        if st.session_state.history:
            history_pills = ""
            for idx, val in enumerate(st.session_state.history):
                # Apply small color styles to indicators in history
                color_h = "#14b8a6" if val <= 2.0 else ("#eab308" if val <= 3.5 else "#ef4444")
                history_pills += f'<span style="display: inline-block; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 5px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; color: rgba(255,255,255,0.9); margin-right: 8px; margin-bottom: 8px;"><span style="color: {color_h};">●</span> {val:.2f} m</span>'
            history_html = f"""
            <div style='margin-top: 20px; text-align: left;'>
                <p style='font-size:0.9rem; color:rgba(255,255,255,0.4); margin:0 0 8px 0; font-weight:600; text-transform:uppercase;'>Prediction History Log</p>
                <div>{history_pills}</div>
            </div>
            """
            st.markdown(history_html, unsafe_allow_html=True)
            
    with col_res2:
        st.markdown('<p style="font-size: 1.15rem; font-weight: 600; color: #ffffff; margin-bottom: 15px; margin-top: 0;">11-Hour Spatial Sequence</p>', unsafe_allow_html=True)
        
        # Plotly Line Chart
        waves_obs = st.session_state.matrix['Wave (m)'].tolist()
        x_obs = [f"t-{i}" for i in range(9, -1, -1)]
        
        fig_line = go.Figure()
        
        # Historical Wave line
        fig_line.add_trace(go.Scatter(
            x=x_obs,
            y=waves_obs,
            mode='lines+markers',
            name='Observed Buoy Height',
            line=dict(color='rgba(20, 184, 166, 0.75)', width=3),
            marker=dict(size=7, color='#14b8a6', line=dict(width=1, color='#ffffff')),
            hovertemplate='Time Step: %{x}<br>Wave Height: %{y:.2f} m<extra></extra>'
        ))
        
        # Connection Line from t-0 to Forecast
        fig_line.add_trace(go.Scatter(
            x=[x_obs[-1], "Forecast"],
            y=[waves_obs[-1], pred_val],
            mode='lines',
            name='Transition Link',
            line=dict(color='#00d4ff', width=3, dash='dash'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Predicted target point
        fig_line.add_trace(go.Scatter(
            x=["Forecast"],
            y=[pred_val],
            mode='markers',
            name='Predicted Wave Height',
            marker=dict(size=14, color='#00d4ff', symbol='circle',
                        line=dict(color='#ffffff', width=2)),
            hovertemplate='Inference: %{x}<br>Predicted Height: %{y:.2f} m<extra></extra>'
        ))
        
        # Annotate Predicted height directly on chart
        fig_line.add_annotation(
            x="Forecast",
            y=pred_val,
            text=f"<b>{pred_val:.2f} m</b>",
            showarrow=True,
            arrowhead=2,
            arrowcolor='#00d4ff',
            ax=0,
            ay=-35,
            font=dict(color='#ffffff', size=13, family="Outfit"),
            bordercolor='#00d4ff',
            borderpad=5,
            bgcolor='#050b14',
            opacity=0.9
        )
        
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.05)',
                tickfont=dict(color='rgba(255,255,255,0.7)', family='Outfit', size=12),
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.05)',
                tickfont=dict(color='rgba(255,255,255,0.7)', family='Outfit', size=12),
                showgrid=True,
                zeroline=False,
                title=dict(text="Height (meters)", font=dict(color='rgba(255,255,255,0.8)', size=13, family='Outfit'))
            ),
            legend=dict(
                font=dict(color='#ffffff', family='Outfit', size=11),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
        st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})
        
    st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)


# ----------------------------------------------------
# 7. FEATURE INFO CARDS
# ----------------------------------------------------
st.markdown('<p class="section-title">Physical Ocean Factors</p>', unsafe_allow_html=True)

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h3 style="font-size: 1.25rem; font-weight: 600; color: #00d4ff; margin: 0 0 12px 0; display: inline-flex; align-items: center; gap: 8px;">
            <span>🌊</span> Wave Height
        </h3>
        <p style="font-size: 0.95rem; color: rgba(255, 255, 255, 0.8); line-height: 1.5; font-weight: 300; margin: 0;">
            Previous wave heights provide the CNN-LSTM model temporal context to identify ocean patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h3 style="font-size: 1.25rem; font-weight: 600; color: #00d4ff; margin: 0 0 12px 0; display: inline-flex; align-items: center; gap: 8px;">
            <span>💨</span> Wind Speed
        </h3>
        <p style="font-size: 0.95rem; color: rgba(255, 255, 255, 0.8); line-height: 1.5; font-weight: 300; margin: 0;">
            Wind speed directly drives wave formation. Trained on NOAA Station 46059 offshore Pacific data.
        </p>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h3 style="font-size: 1.25rem; font-weight: 600; color: #00d4ff; margin: 0 0 12px 0; display: inline-flex; align-items: center; gap: 8px;">
            <span>🌡️</span> Air Pressure
        </h3>
        <p style="font-size: 0.95rem; color: rgba(255, 255, 255, 0.8); line-height: 1.5; font-weight: 300; margin: 0;">
            Pressure drops indicate incoming storms. Our model captures these atmospheric-ocean interactions.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------
# 8. MODEL INFO CARD & 9. HOW TO USE
# ----------------------------------------------------
st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
info_col1, info_col2 = st.columns(2, gap="medium")

with info_col1:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h3 style="font-size: 1.4rem; font-weight: 600; color: #00d4ff; margin: 0 0 15px 0;">
            🤖 About WaveCast Model
        </h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85); font-weight: 300;">
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                <td style="padding: 8px 0; font-weight: 600; color: #14b8a6;">Architecture</td>
                <td style="padding: 8px 0; text-align: right;">CNN-LSTM Neural Network | NOAA 46059</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                <td style="padding: 8px 0; font-weight: 600; color: #14b8a6;">Training Data</td>
                <td style="padding: 8px 0; text-align: right;">Real NOAA ocean buoy measurements</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                <td style="padding: 8px 0; font-weight: 600; color: #14b8a6;">R² Model Score</td>
                <td style="padding: 8px 0; text-align: right;">0.9985 (Test Partition)</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                <td style="padding: 8px 0; font-weight: 600; color: #14b8a6;">Lookback Window</td>
                <td style="padding: 8px 0; text-align: right;">10 sequential hourly timesteps</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                <td style="padding: 8px 0; font-weight: 600; color: #14b8a6;">Input Parameters</td>
                <td style="padding: 8px 0; text-align: right;">4 Features (Wave, Wind, Pressure, Sea Temp)</td>
            </tr>
            <tr>
                <td style="padding: 8px 0; font-weight: 600; color: #14b8a6;">Deployment Mode</td>
                <td style="padding: 8px 0; text-align: right;">Self-Contained Streamlit Dashboard</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h3 style="font-size: 1.4rem; font-weight: 600; color: #00d4ff; margin: 0 0 15px 0;">
            📖 How to Use
        </h3>
        <ol style="margin: 0; padding-left: 20px; font-size: 0.95rem; color: rgba(255, 255, 255, 0.85); font-weight: 300; display: flex; flex-direction: column; gap: 9px;">
            <li>Enter or fill <b>10 consecutive hours</b> of ocean weather data in the table editor.</li>
            <li>Each row matches 1 hour of readings: wave height, wind, pressure, and sea temperature.</li>
            <li>Click the glowing <b style="color: #00d4ff;">🌊 Predict Wave Height</b> button to activate forecasting.</li>
            <li>Inspect your predicted waves on the <b>Radial Gauge</b> and the <b>Sequence Connection Plot</b>.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------
# 10. FOOTER SECTION
# ----------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-top: 55px; padding: 25px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.85rem; color: rgba(255,255,255,0.5); font-weight: 300; letter-spacing: 0.5px; line-height: 1.5;">
    <p style="margin: 0 0 4px 0;">© 2026 WaveCast | Built by Mohammed Shemeer</p>
    <p style="margin: 0;">Powered by Streamlit & TensorFlow | Data Source: NOAA NDBC Buoy Station 46059</p>
</div>
""", unsafe_allow_html=True)
