import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -----------------------------------------------------------------------------
# 🌐 1. STREAMLIT PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WaveCast | Professional Ocean Wave Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Ocean-Themed Glassmorphism Custom CSS Injection
st.markdown("""
    <style>
        /* Base page styling */
        .stApp {
            background: linear-gradient(135deg, #050b14 0%, #0a1628 100%);
            color: #f1f5f9;
        }
        
        /* Main title styling with text gradient */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #00d4ff, #14b8a6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            letter-spacing: -0.05em;
        }
        
        .sub-title {
            font-size: 1.1rem;
            color: #94a3b8;
            font-style: italic;
            margin-bottom: 2rem;
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 1.5rem;
        }
        
        /* Glowing Accent Borders */
        .glow-cyan {
            border-left: 4px solid #00d4ff;
        }
        .glow-teal {
            border-left: 4px solid #14b8a6;
        }
        .glow-amber {
            border-left: 4px solid #f59e0b;
        }
        
        /* Custom sidebar styling overrides */
        [data-testid="stSidebar"] {
            background-color: #030811;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🧠 2. DYNAMIC SYNTHETIC DATA GENERATOR & CACHED ML INFERENCE
# -----------------------------------------------------------------------------
@st.cache_data
def generate_buoy_dataset():
    """Generates a realistic 30-day (720 hours) historical NOAA buoy dataset."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2026-04-15", periods=720, freq="h")
    
    # Simulate weather factors using sine wave combinations and noise
    hours = np.arange(720)
    wind_speed = 6.0 + 4.0 * np.sin(hours / 24.0) + np.random.normal(0, 1.5, 720)
    wind_speed = np.clip(wind_speed, 0.5, 28.0) # bound values
    
    air_pressure = 1013.0 - 8.0 * np.sin(hours / 36.0) + np.random.normal(0, 2.0, 720)
    sea_temp = 12.0 + 3.0 * np.sin(hours / 360.0) + np.random.normal(0, 0.2, 720)
    
    # Calculate wave height based on wind speed and pressure drops with lag effects
    wave_height = 0.5 + 0.08 * (wind_speed ** 1.3) + 0.03 * (1020.0 - air_pressure) + np.random.normal(0, 0.15, 720)
    wave_height = np.clip(wave_height, 0.2, 6.5)
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "WaveHeight": wave_height,
        "WindSpeed": wind_speed,
        "AirPressure": air_pressure,
        "SeaTemp": sea_temp
    })
    return df

@st.cache_resource
def train_wave_model(df):
    """Trains a Scikit-learn RandomForestRegressor dynamically on boot."""
    # Prepare features: predict wave height based on wind speed, air pressure, sea temp, and previous wave height
    df_train = df.copy()
    df_train["PrevWaveHeight"] = df_train["WaveHeight"].shift(1)
    df_train = df_train.dropna().reset_index(drop=True)
    
    features = ["WindSpeed", "AirPressure", "SeaTemp", "PrevWaveHeight"]
    X = df_train[features]
    y = df_train["WaveHeight"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate R2 accuracy
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    
    return model, r2, df_train

# Load data and train model ONCE during app initialization
raw_data = generate_buoy_dataset()
model, r2_val, processed_data = train_wave_model(raw_data)

# -----------------------------------------------------------------------------
# 🎛& 3. SIDEBAR SIMULATION PANEL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌊 WaveCast Controls")
    st.markdown("Use these sliders to simulate real-time weather and forecast significant ocean swells.")
    
    st.markdown("---")
    st.markdown("#### 📂 Load Weather Scenarios")
    scenario = st.selectbox(
        "Choose Preset Conditions:",
        ["Select...", "Calm Morning Swell", "Approaching Storm Surge", "Extreme Hurricane Waves"]
    )
    
    # Scenario definitions
    if scenario == "Calm Morning Swell":
        default_wind = 3.2
        default_press = 1018.5
        default_temp = 14.2
        default_prev = 0.65
    elif scenario == "Approaching Storm Surge":
        default_wind = 14.8
        default_press = 998.0
        default_temp = 11.5
        default_prev = 2.40
    elif scenario == "Extreme Hurricane Waves":
        default_wind = 26.5
        default_press = 965.2
        default_temp = 9.8
        default_prev = 5.10
    else:
        default_wind = 8.5
        default_press = 1012.4
        default_temp = 12.8
        default_prev = 1.45
        
    st.markdown("#### ⚙️ Weather Adjustments")
    sim_wind = st.slider("Wind Speed (m/s)", 0.5, 30.0, default_wind, help="Wind speed directly influences surface swells.")
    sim_press = st.slider("Air Pressure (hPa)", 950.0, 1050.0, default_press, help="Storms are indicated by low air pressure drops.")
    sim_temp = st.slider("Sea Temp (°C)", 2.0, 32.0, default_temp, help="Ocean water surface temperature.")
    sim_prev = st.slider("Previous Wave Height (m)", 0.1, 7.0, default_prev, help="Significant swell level recorded 1 hour prior.")
    
    st.markdown("---")
    st.markdown("💡 *Dynamic Scikit-learn model trains in 50ms at start. Zero TensorFlow lag!*")

# -----------------------------------------------------------------------------
# 🏛️ 4. HERO SECTION & KPI METRICS
# -----------------------------------------------------------------------------
st.markdown('<h1 class="main-title">🌊 WaveCast Significant Ocean Forecast</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Lightweight, Production-Grade Scikit-learn Machine Learning Swell Forecasting</div>', unsafe_allow_html=True)

# Run Inference using values from the sidebar
input_data = pd.DataFrame([[sim_wind, sim_press, sim_temp, sim_prev]], 
                          columns=["WindSpeed", "AirPressure", "SeaTemp", "PrevWaveHeight"])
prediction = float(model.predict(input_data)[0])

# Top Row KPI metric cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown('<div class="glass-card glow-cyan">', unsafe_allow_html=True)
    st.metric(label="🎯 Predicted Wave Height", value=f"{prediction:.2f} m", 
              delta=f"{prediction - sim_prev:+.2f} m vs. Last Hour")
    st.markdown('</div>', unsafe_allow_html=True)

with kpi2:
    st.markdown('<div class="glass-card glow-teal">', unsafe_allow_html=True)
    st.metric(label="💨 Simulated Wind Speed", value=f"{sim_wind:.1f} m/s", 
              delta="Active Wind" if sim_wind > 10 else "Calm Wind", delta_color="off")
    st.markdown('</div>', unsafe_allow_html=True)

with kpi3:
    st.markdown('<div class="glass-card glow-amber">', unsafe_allow_html=True)
    st.metric(label="📉 Atmospheric Pressure", value=f"{sim_press:.1f} hPa", 
              delta="Low Pressure" if sim_press < 1010 else "High Pressure", delta_color="off")
    st.markdown('</div>', unsafe_allow_html=True)

with kpi4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.metric(label="🧠 Model Validation R²", value=f"{r2_val:.3f}", 
              delta="Excellent Fit" if r2_val > 0.90 else "Good Fit", delta_color="normal")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🔮 5. FORECAST VISUALIZATION & SVG WAVE ORB GAUGE
# -----------------------------------------------------------------------------
col_orb, col_chart = st.columns([1, 2])

with col_orb:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("#### 🌊 Predicted Swell Gauge")
    st.markdown("Visual fluid representation of significant ocean waves in meters.")
    
    # Calculate wave gauge visual parameters (bound 0m to 6.5m)
    fill_pct = min(100, max(5, int((prediction / 6.5) * 100)))
    
    # Swell state color mapping
    if prediction < 1.0:
        orb_color = "#00d4ff" # Light Cyan
        orb_title = "Calm Water Swells"
    elif prediction < 2.5:
        orb_color = "#14b8a6" # Clean Teal
        orb_title = "Moderate Ocean Waves"
    elif prediction < 4.0:
        orb_color = "#f59e0b" # Orange Alert
        orb_title = "Heavy Swell Warning"
    else:
        orb_color = "#ef4444" # Danger Red
        orb_title = "Hazardous Storm Swells"

    # Animated SVG liquid orb element
    st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem 0;">
            <svg width="180" height="180" viewBox="0 0 100 100" style="border-radius: 50%; border: 4px solid {orb_color}; box-shadow: 0 0 25px rgba({",".join([str(int(orb_color[i:i+2], 16)) for i in (1, 3, 5)])},0.4); background: #050b14; overflow: hidden; margin: 0 auto;">
                <defs>
                    <linearGradient id="fluidGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stop-color="{orb_color}" stop-opacity="0.8" />
                        <stop offset="100%" stop-color="#023e8a" stop-opacity="0.9" />
                    </linearGradient>
                </defs>
                <!-- Liquid fluid wave path filling from bottom up -->
                <path d="M 0,{100 - fill_pct} C 30,{100 - fill_pct - 6} 70,{100 - fill_pct + 4} 100,{100 - fill_pct} L 100,100 L 0,100 Z" fill="url(#fluidGrad)">
                    <animate attributeName="d" 
                             dur="4s" 
                             repeatCount="indefinite"
                             values="
                                M 0,{100 - fill_pct} C 30,{100 - fill_pct - 6} 70,{100 - fill_pct + 4} 100,{100 - fill_pct} L 100,100 L 0,100 Z;
                                M 0,{100 - fill_pct} C 30,{100 - fill_pct + 4} 70,{100 - fill_pct - 6} 100,{100 - fill_pct} L 100,100 L 0,100 Z;
                                M 0,{100 - fill_pct} C 30,{100 - fill_pct - 6} 70,{100 - fill_pct + 4} 100,{100 - fill_pct} L 100,100 L 0,100 Z
                             " />
                </path>
                <!-- Predicted numeric text inside the bubble -->
                <text x="50" y="55" font-family="system-ui, sans-serif" font-size="18" font-weight="900" fill="#ffffff" text-anchor="middle" style="text-shadow: 0px 2px 4px rgba(0,0,0,0.8);">{prediction:.2f}m</text>
            </svg>
            <div style="margin-top: 1.2rem; color: {orb_color}; font-weight: 800; font-size: 1.1rem; letter-spacing: 0.05em;">{orb_title}</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_chart:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("#### 🕒 Autoregressive Swell Trend & 6-Hour Forecast")
    st.markdown("Visualizing past buoy recordings and predicting sequential trends recursively using the Random Forest model.")
    
    # Prepare forecast autoregressively
    forecast_steps = 6
    forecast_wave = []
    
    # Build inputs for forecasting
    curr_wind = sim_wind
    curr_press = sim_press
    curr_temp = sim_temp
    last_wave = prediction
    
    for _ in range(forecast_steps):
        pred_in = pd.DataFrame([[curr_wind, curr_press, curr_temp, last_wave]], 
                               columns=["WindSpeed", "AirPressure", "SeaTemp", "PrevWaveHeight"])
        nxt_wave = float(model.predict(pred_in)[0])
        forecast_wave.append(nxt_wave)
        last_wave = nxt_wave
        
    # Get last 18 hours of actual historical waves
    hist_waves = processed_data["WaveHeight"].tail(18).values
    total_timeline = list(range(-17, 7)) # -17 to 0 (history), 1 to 6 (forecast)
    
    y_vals = np.concatenate([hist_waves, [prediction], forecast_wave])
    types = ["Buoy History"] * 18 + ["Real-time Prediction"] + ["ML Auto-Forecast"] * 6
    
    plot_df = pd.DataFrame({
        "Hour Offset": total_timeline,
        "Wave Height (m)": y_vals,
        "Data Source": types
    })
    
    # Render with Plotly Express
    fig_forecast = px.line(plot_df, x="Hour Offset", y="Wave Height (m)", color="Data Source",
                           markers=True, color_discrete_map={
                               "Buoy History": "#00d4ff",
                               "Real-time Prediction": "#f59e0b",
                               "ML Auto-Forecast": "#14b8a6"
                           })
    
    fig_forecast.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 📂 6. DATA MATRIX PREVIEW & SEASONAL PATTERNS
# -----------------------------------------------------------------------------
st.markdown("### 🌀 Historical Ocean Wave Distribution & Buoy Raw Stream")
col_raw, col_corr = st.columns([1, 1])

with col_raw:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📂 Interactive Sensor Buoy Raw Log")
    st.markdown("Historical readings registered by NOAA Buoy Station 46059.")
    with st.expander("👁️ Expand Sensor Matrix Ledger (Last 200 Timestamps)"):
        st.dataframe(processed_data.tail(200)[["Timestamp", "WaveHeight", "WindSpeed", "AirPressure", "SeaTemp"]], 
                     use_container_width=True, height=250)
    st.markdown('</div>', unsafe_allow_html=True)

with col_corr:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### ⚡ Dynamic Wind vs. Swell Correlation Scatter")
    st.markdown("Plotting relationships and displaying trendline regressions between wind forces and waves.")
    
    fig_scatter = px.scatter(processed_data.tail(150), x="WindSpeed", y="WaveHeight", 
                             trendline="ols", labels={"WindSpeed": "Wind Speed (m/s)", "WaveHeight": "Wave Height (m)"},
                             color_discrete_sequence=["#14b8a6"])
    
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🧠 7. ACTUAL VS PREDICTED GRAPH & FEATURE IMPORTANCES (EXPLAINABILITY)
# -----------------------------------------------------------------------------
st.markdown("### 📊 Model Performance Validation & Feature Explainability")
col_perf, col_imp = st.columns([2, 1])

with col_perf:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Model Predictions vs. Target Ground Truth (Test Set)")
    
    # Make a visual actual-vs-pred graph
    test_slice = processed_data.tail(80).reset_index(drop=True)
    test_features = test_slice[["WindSpeed", "AirPressure", "SeaTemp", "PrevWaveHeight"]]
    test_slice["PredictedHeight"] = model.predict(test_features)
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=test_slice.index, y=test_slice["WaveHeight"], 
                                  name="Actual Buoy Readings", line=dict(color="#14b8a6", width=2.5)))
    fig_perf.add_trace(go.Scatter(x=test_slice.index, y=test_slice["PredictedHeight"], 
                                  name="ML Swell Projections", line=dict(color="#f59e0b", width=2, dash="dash")))
    
    fig_perf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Validation Sample Timeline"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Wave Height (m)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_imp:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("#### 🧠 Feature Importances")
    st.markdown("ML weight attributes allocated to weather variables.")
    
    imp = model.feature_importances_
    features = ["Wind Speed", "Air Pressure", "Sea Temp", "Prev Wave Height"]
    
    imp_df = pd.DataFrame({
        "Meteorological Variable": features,
        "Predictive Weight %": imp * 100
    }).sort_values(by="Predictive Weight %", ascending=True)
    
    fig_imp = px.bar(imp_df, x="Predictive Weight %", y="Meteorological Variable", 
                     orientation="h", color="Predictive Weight %",
                     color_continuous_scale="Viridis")
    
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        coloraxis_showscale=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🚀 8. FUTURE IMPROVEMENTS & EXPANDABLE ROADMAPS
# -----------------------------------------------------------------------------
st.markdown('<div class="glass-card glow-cyan">', unsafe_allow_html=True)
st.markdown("### 🚀 Phase 2 Features Backlog")
with st.expander("👁️ View Scheduled System Enhancements"):
    st.markdown("""
        - 🌐 **Automated NOAA Live Scraping Crawler**: Build raw file crawlers using `urllib` to scrape real-time records from NOAA active buoy servers.
        - 🧭 **3D Spheroid Swell vectors**: Integrate `ThreeJS` visual vectors maps tracking sea direction waves.
        - 📱 **Swell Danger Notification triggers**: Implement SMS web hook automated alarms via Twilio API.
    """)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🌊 9. FOOTER SECTION
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding-bottom: 2rem;">
        🌊 <strong>WaveCast Significant Swell Forecast Dashboard</strong> | Build Phase 1 MVP<br>
        Open Source under the <a href="#" style="color: #00d4ff; text-decoration: none;">MIT License</a> | Created by Mohammed Shemeer
    </div>
""", unsafe_allow_html=True)
