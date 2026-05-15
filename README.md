# 🌊 WaveCast: Ocean Wave Height Prediction System

**WaveCast** is a production-ready, fully responsive, self-contained Streamlit web application that predicts ocean wave heights. It uses a high-precision deep learning **CNN-LSTM** sequence model trained on real-world ocean telemetry from **NOAA NDBC Buoy Station 46059** in the offshore Pacific.

The dashboard integrates interactive tables, safety analytics, visual gauges, historical transition line plots, and premium styled animated backgrounds that recreate a deep ocean atmosphere.

---

## 🚀 Key Features

*   **Premium Visual Design**: Styled in a custom **Deep Ocean Dark Theme** with harmonic cyan and teal accents, glassmorphic floating cards, floating bubble animations, SVG shifting waves, and drifting particulate lights.
*   **Dual Operating Modes**:
    *   **Inference Mode**: Loads a tensorflow-based `CNN-LSTM` model and joblib scaling assets from the `models/` directory, predicting with $R^2 = 0.9985$ accuracy.
    *   **Demo Mode**: In the absence of physical model checkpoints, dynamically switches to a robust, physics-grounded deterministic ocean formula using observed waves, gust velocities, barometric transitions, and sea temperature anomalies.
*   **Interactive Input Matrix**: Allows quick modifications of 10 consecutive hourly steps across 4 core telemetry variables using a styled `st.data_editor`.
*   **Safety Threshold Gauge**: Integrates a radial Plotly dial showing Safe ($0-2.0\text{m}$), Moderate ($2.0-3.5\text{m}$), and Dangerous ($>3.5\text{m}$) maritime safety flags.
*   **Forecast Chart**: Visualizes past observations connected via a transition bridge directly to the predicted next-hour forecast point.

---

## 🛠️ File Structure

```
wavecast/
├── app.py                  # Streamlit application source (styling, logic & UI)
├── requirements.txt        # Production dependency specifications
├── README.md               # Implementation documentation
└── models/                 # Deep learning checkpoints (optional)
    ├── wave_cnn_lstm.keras # Compiled Keras model
    ├── x_scaler.pkl        # Input feature scaler
    └── y_scaler.pkl        # Output target scaler
```

---

## 📦 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9 - 3.11 installed.

### 2. Installation
Clone the workspace files and install all dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launching the App
Run the following command to start the Streamlit server:
```bash
streamlit run app.py
```
Open the provided local URL (typically `http://localhost:8501`) in your browser to view the dashboard!

---

## 🤖 Model Architecture Details
The system processes sequential multidimensional timesteps ($10 \text{ hours} \times 4 \text{ features}$) to capture both high-frequency spatial gradients and low-frequency long-term dependencies:
1.  **CNN Layers**: Detect local temporal correlation boundaries across wind speed, pressure, temperature, and sea peaks.
2.  **LSTM Layers**: Retain memory trends to model wave propagation profiles.
3.  **Dense Layers**: Output the scaled next-hour wave height in meters.

### Data Attributes
*   **Wave Height (m)**: Dominant crest-to-trough sequence.
*   **Wind Speed (m/s)**: Primary momentum transfer vector forming waves.
*   **Air Pressure (hPa)**: Storm cycle indicator. Drops signify weather fronts and surges.
*   **Sea Temperature (°C)**: Thermodynamic index affecting marine energy transfers.

---

*Built with ❤️ by Mohammed Shemeer*
