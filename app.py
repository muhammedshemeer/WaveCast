import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# 🌐 1. PAGE CONFIGURATION & TITLE
# ==========================================
st.set_page_config(
    page_title="WaveCast | Minimal Ocean Dashboard",
    page_icon="🌊",
    layout="wide"
)

# Header Section
st.title("🌊 WaveCast: Significant Ocean Swell Dashboard")
st.markdown("A lightweight, beginner-friendly ocean swell visualization and forecasting tool.")
st.markdown("---")

# ==========================================
# 🎛️ 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Simulation Panel")
st.sidebar.markdown("Adjust the sliders below to simulate changing meteorological parameters:")

# Slide control for Wind Speed
sim_wind = st.sidebar.slider(
    label="Wind Speed (m/s)",
    min_value=0.5,
    max_value=25.0,
    value=8.5,
    step=0.5,
    help="Higher wind speeds generate larger significant wave heights."
)

# Slide control for Air Pressure
sim_press = st.sidebar.slider(
    label="Barometric Pressure (hPa)",
    min_value=960,
    max_value=1040,
    value=1013,
    step=1,
    help="Low pressure drops indicate incoming ocean storms."
)

# ==========================================
# 💾 3. SAMPLE DATASET GENERATION
# ==========================================
@st.cache_data
def generate_sample_data():
    """Generates 24 hours of synthetic buoy ocean logs using NumPy and Pandas."""
    np.random.seed(42)
    hours = list(range(1, 25))
    
    # Generate smooth wave trend using sine waves + random wind fluctuations
    base_waves = np.sin(np.linspace(0, 5, 24)) * 0.5 + 1.5
    noise = np.random.normal(0, 0.15, 24)
    wave_heights = np.clip(base_waves + noise, 0.3, 5.0)
    
    # Pack into a Pandas DataFrame
    df = pd.DataFrame({
        "Hour Offset": [f"-{24 - h}h" for h in hours],
        "Wave Height (m)": wave_heights,
        "Wind Speed (m/s)": np.linspace(6.0, 11.0, 24) + np.random.normal(0, 0.5, 24)
    })
    return df

# Load the historical dataset
buoy_data = generate_sample_data()

# ==========================================
# 🔮 4. INTERACTIVE PHYSICS-BASED INFERENCE
# ==========================================
# Predict swell size dynamically: Swell size correlates with wind shear and pressure drop
predicted_height = 0.5 + (0.12 * sim_wind) + (0.015 * (1020 - sim_press))
predicted_height = max(0.2, min(7.5, predicted_height)) # Bound values safely

# ==========================================
# 📊 5. KPI METRICS & DISPLAY CARDS
# ==========================================
# Create two professional-looking metrics cards side-by-side
kpi_col1, kpi_col2 = st.columns(2)

with kpi_col1:
    st.metric(
        label="🎯 Predicted Wave Height",
        value=f"{predicted_height:.2f} meters",
        delta=f"{(predicted_height - 1.5):+.2f} m vs. Base Swell"
    )

with kpi_col2:
    st.metric(
        label="💨 Simulated Wind Speed",
        value=f"{sim_wind:.1f} m/s",
        delta="High Swell Alert" if sim_wind > 12.0 else "Calm Seas"
    )

st.markdown("---")

# ==========================================
# 📈 6. PLOTLY GRAPH & DATA VIEW
# ==========================================
chart_col, data_col = st.columns([2, 1])

with chart_col:
    st.subheader("🕒 Historical Buoy Swell Trend (Last 24 Hours)")
    
    # Render line chart with Plotly Express
    fig = px.line(
        buoy_data,
        x="Hour Offset",
        y="Wave Height (m)",
        markers=True,
        title="BUOY STATION 46059 - SIGNIFICANT WAVE HEIGHTS",
        color_discrete_sequence=["#00d4ff"] # Cyan
    )
    
    # Visual updates to Plotly theme to blend with dark dashboard
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time Offset",
        yaxis_title="Height (meters)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

with data_col:
    st.subheader("📂 raw buoy sensors log")
    st.markdown("Expand the widget below to preview raw tabular recordings.")
    
    # Neat expandable raw data frame ledger
    with st.expander("👁️ Preview Raw Matrix Ledger"):
        st.dataframe(buoy_data, use_container_width=True, height=270)

# ==========================================
# 🌊 7. FOOTER SECTION
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "🌊 <strong>WaveCast Significant Swell Dashboard</strong> | Phase 1 Minimal MVP<br>"
    "Open Source MIT License | Developed with Streamlit and Plotly"
    "</div>",
    unsafe_allow_html=True
)
