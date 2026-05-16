# <p align="center">🌊 WaveCast — CNN-LSTM Ocean Wave Height Prediction System</p>

<p align="center">
  <strong>A high-precision deep learning forecasting system trained on real-world NOAA telemetry to predict maritime conditions.</strong>
</p>

<p align="center">
  <a href="https://wavecast-erfq2rjs4cmdrjf7qjjzaw.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Launch%20WaveCast-00D4FF?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" />
  </a>
</p>

<p align="center">
  <a href="https://wavecast-erfq2rjs4cmdrjf7qjjzaw.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App" /></a>
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange.svg" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
  <img src="https://img.shields.io/badge/R²%20Score-0.9985-brightgreen.svg" alt="R² Score" />
</p>

---

## 📸 Dashboard Preview

<p align="center">
  <img src="[SCREENSHOT_1]" width="48%" alt="Wave Orb Gauge Display" />
  <img src="[SCREENSHOT_2]" width="48%" alt="Hero Dashboard Interface" />
</p>

<p align="center">
  <em>
    <strong>Left:</strong> 🌊 Wave Orb Gauge — Real-time animated prediction display with Moderate Sea State safety assessment. <br/>
    <strong>Right:</strong> 🏠 Hero Dashboard — CNN-LSTM Ocean Forecast System with animated wave interface.
  </em>
</p>

---

## ✨ Key Features

*   🧠 **CNN-LSTM Deep Learning Architecture**: Leverages sequential temporal patterns to capture spatial-temporal gradients across buoy telemetry.
*   📡 **Real NOAA NDBC Buoy Data**: Trained on real-world ocean telemetry from **Station 46059** (Pacific Ocean, offshore California).
*   🎯 **High-Accuracy Predictions**: Achieves an exceptional **$R^2 = 0.9985$** on the test dataset.
*   🌊 **Animated Water Wave Orb**: A custom HTML/CSS/JS Circular Wave Orb that dynamically fills based on predicted heights, featuring out-of-phase overlapping sine wave animations.
*   🚦 **Smart Safety Assessments**: Instant, color-coded classification showing **SAFE** 🟢, **MODERATE** 🟡, or **DANGEROUS** 🔴 sea states with guidance for maritime operators.
*   📊 **Interactive 11-Hour Sequence Chart**: An interactive Plotly chart mapping the transition from 10 observed sequence hours to the projected next-hour forecast.
*   🎨 **Premium Deep Ocean Theme**: Styled in a custom dark navy aesthetic with glassmorphic cards, rising bubbles, drifting light particles, and looping horizontal SVG wave backdrops.
*   ⚡ **Robust Demo Mode**: Fully functional with a deterministic physical formula fallback in case physical model checkpoints are absent.
*   📱 **Fully Responsive Layout**: Built with absolute responsiveness, allowing seamless operation across desktop, tablet, and mobile browsers.

---

## 🔬 Model & Training Details

The system uses a sequential deep neural network combining 1D Convolutional layers (CNN) with Long Short-Term Memory units (LSTM):

| Component | Details |
| :--- | :--- |
| **Neural Architecture** | CNN-LSTM (Conv1D ➔ MaxPooling1D ➔ LSTM ➔ Dense) |
| **Training Dataset** | NOAA National Data Buoy Center (NDBC) Station 46059 |
| **Prediction Target** | Next-Hour Ocean Wave Height (meters) |
| **Lookback Window** | 10 consecutive hours ($t-9$ to $t-0$) |
| **Accuracy Metric ($R^2$)** | **0.9985** |
| **Production Server** | Streamlit Community Cloud |

---

## 📡 Input Features

WaveCast uses 4 core telemetry features measured hourly to forecast swell propagation:

1.  🌊 **Wave Height (m)**: The dominant wave crest-to-trough sequence, serving as the temporal base.
2.  💨 **Wind Speed (m/s)**: The main physical driver behind wave energy transfer, wind currents, and swell growth.
3.  🌡️ **Sea Temperature (°C)**: Thermophysical indicator showing water density trends and thermal current energy cycles.
4.  🎈 **Air Pressure (hPa)**: Barometric pressure patterns; sharp pressure drops indicate storm fronts and rising storm surges.

---

## ⚙️ How It Works

1.  📥 **Data Input**: The user loads high-frequency telemetry sequences into the custom Streamlit `st.data_editor` or clicks **Fill Sample Data** to load real ocean buoy patterns.
2.  🔄 **Preprocessing & Scaling**: The 10-hour feature matrices are structured into standard shapes and normalized using pre-trained `StandardScaler` transformations.
3.  🧠 **Neural Network Inference**: The CNN-LSTM network parses temporal swell patterns, outputting the forecasted wave height.
4.  🚦 **Safety Classification**: The model translates forecasted wave heights into clear classifications:
    *   **SAFE** ($H \le 2.0\text{m}$): Calm, optimal marine conditions.
    *   **MODERATE** ($2.0\text{m} < H \le 3.5\text{m}$): Sweeping swells; caution is highly advised.
    *   **DANGEROUS** ($H > 3.5\text{m}$): Massive seas; small vessels must avoid transit.
5.  🎨 **Dynamic Rendering**: Visual layouts (the Water Wave Orb, the Plotly forecast graph, and history logs) auto-update instantly.

---

## 📂 File Structure

```text
wavecast/
├── app.py                  # Self-contained Streamlit Web Application (UI, CSS, and logic)
├── requirements.txt        # Lightweight production dependency configuration
├── README.md               # Professional project overview & walkthrough
├── LICENSE                 # MIT License details
└── models/                 # Deep learning checkpoints (optional)
    ├── wave_cnn_lstm.keras # Saved CNN-LSTM neural net model
    ├── x_scaler.pkl        # Serialized input scaler (MinMax/Standard)
    └── y_scaler.pkl        # Serialized target output scaler
```

---

## 🛠️ Getting Started

Follow these steps to set up and run the WaveCast dashboard locally on your machine:

### 1. Prerequisites
- Python 3.9 - 3.11 installed.
- Git (for repository cloning).

### 2. Installation
Clone the project repository and navigate into the workspace directory:
```bash
git clone https://github.com/muhammedshemeer/WaveCast.git
cd WaveCast
```
Install the lightweight dependency requirements:
```bash
pip install -r requirements.txt
```

### 3. Run Locally
Launch the local Streamlit development server:
```bash
streamlit run app.py
```
Open your web browser and navigate to **`http://localhost:8501`** to view the live dashboard!

*(Optional: Place your pre-trained `wave_cnn_lstm.keras`, `x_scaler.pkl`, and `y_scaler.pkl` files into a `models/` directory in the root of the project to enable full AI Inference Mode. If left empty, the app will run seamlessly in Demo Mode.)*

---

## 💻 Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/streamlit-%23FF4B4B.svg?style=flat&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=flat&logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/joblib-blue?style=flat" alt="Joblib" />
</p>

---

## 🚀 Live Demo

Experience the full live dashboard directly in your web browser:

<p align="center">
  <a href="https://wavecast-erfq2rjs4cmdrjf7qjjzaw.streamlit.app/">
    <img src="https://img.shields.io/badge/🌊%20LAUNCH%20WAVECAST-CLICK%20HERE-00d4ff?style=for-the-badge&logo=launchpad&logoColor=white" alt="Launch Live App" />
  </a>
</p>

---

## 👨‍💻 Author

**Mohammed Shemeer**
*   **LinkedIn**: [Mohammed Shemeer](https://linkedin.com/in/mohammed-shemeer-aiml)
*   **GitHub**: [@muhammedshemeer](https://github.com/muhammedshemeer)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
