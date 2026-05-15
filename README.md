# 🌊 WaveCast: Premium CNN-LSTM Ocean Wave Prediction System

WaveCast is a production-grade, highly immersive, ocean-themed Streamlit dashboard powered by a hybrid **CNN-LSTM deep learning model**. It predicts significant wave heights based on meteorological features (Wind Speed, Air Pressure, and Temperature) using sliding window time-series sequences. 

The application is completely self-contained in Python with zero external API dependencies, featuring a high-performance cached neural network engine and a gorgeous deep-ocean dark themed glassmorphic UI.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![R2 Score](https://img.shields.io/badge/R²%20Score-0.96-brightgreen?style=for-the-badge)

[View Codebase](https://github.com/muhammedshemeer/Wave_prediction) · [Author's LinkedIn](https://linkedin.com/in/mohammed-shemeer-aiml)

</div>

---

## 📈 Model Performance & Graphical Visualizations

| Actual vs Predicted Wave Heights | CNN-LSTM Model Learning Curve |
|:---:|:---:|
| ![Actual vs Predicted](prediction_plot.png) | ![Model Learning Curve](learning_curve.png) |

---

## 🌟 Key Features

*   **🎨 Premium Glassmorphic UI**: Immersive, deep-ocean dark themed interface (`#050b14` to `#0a1628` background) custom-styled with responsive frosted card layouts (`rgba(255,255,255,0.03)`), glowing teal borders, and custom typography.
*   **🔮 High-Performance Neural Engine**: Fast, cached loading of the local Keras model (`wave_prediction_model.h5`) and MinMaxScaler pickles using Streamlit's `@st.cache_resource` for low-latency inference.
*   **🏄 Dynamic Wave Orb Height Gauge**: Beautiful custom SVG wave orb indicator that dynamically fills with animated fluid layers proportional to the predicted ocean swell height (scaled from 0m to 6m).
*   **🛠️ Interactive Sequence Editor**: Built-in `st.data_editor` sequence table allowing the user to manually edit meteorological parameters or quickly populate pre-configured physical curves (Swell/Storm/Calm) with the click of a button.
*   **🛡️ Robust Physics Fallback**: A resilient deterministic physics swell formula is integrated to simulate forecasts based on wind shear and pressure drop if the machine-learning stack (TensorFlow) is unavailable, ensuring high application uptime.
*   **📊 Immersive Plotly Forecasts**: Dynamic multi-line graphs highlighting both historical buoy observations and upcoming wave height projections.

---

## 🧠 CNN-LSTM Neural Network Architecture

To capture both spatial features within meteorological readings and temporal correlations across sequences, the model employs a hybrid deep learning architecture:

```
Input (10 time steps × 4 features)
        ↓
    Conv1D (64 filters, kernel=2, activation=ReLU)
        ↓
    MaxPooling1D (pool_size=2)
        ↓
    LSTM (50 units, activation=tanh)
        ↓
    Dense (1 unit)
        ↓
Output: Predicted Wave Height (Significant Swell in Meters)
```

### Model Performance Metrics:
- **R² Score**: **0.96** on test datasets.
- **Data Source**: Custom meteorological sequences compiled from NOAA NDBC Station 46059 (2018–2023).

---

## 📁 Repository Directory Structure

```
Wave_prediction/
├── streamlit_app/           # Unified Streamlit Application Code
│   ├── app.py              # Main dashboard script (UI, styles, and cached inference)
│   ├── requirements.txt    # Streamlit app direct dependencies
│   ├── README.md           # Application execution guide
│   └── models/             # Local ML models and scalers for deployment
│       ├── wave_prediction_model.h5  # Pre-compiled CNN-LSTM model
│       ├── x_scaler.pkl              # Pickled features scaler
│       └── y_scaler.pkl              # Pickled target scaler
├── data/                    # Dataset storage
│   ├── raw/                 # Raw downloaded NOAA weather tables
│   └── processed/           # Processed sequence arrays
├── models/                  # Root model store (original files)
│   ├── wave_prediction_model.h5
│   ├── feature_scaler.pkl
│   └── target_scaler.pkl
├── download_data.py         # Automated buoy data downloader
├── step2_dataset.py         # Merges raw datasets and handles gaps
├── step3_preprocessing.py   # Compiles sequential matrices and normalizations
├── wave_prediction.py       # Trains, validates, and evaluates the CNN-LSTM network
├── Dockerfile               # Production Docker container definition
├── requirements.txt         # Root requirements mapping
└── LICENSE                  # Project licensing details (MIT)
```

---

## 🚀 Getting Started

### Method 1: Local Deployment

```bash
# 1. Clone the repository
git clone https://github.com/muhammedshemeer/Wave_prediction.git
cd Wave_prediction

# 2. Activate the virtual environment
venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run streamlit_app/app.py

# 5. Open http://localhost:8501 in your browser!
```

### Method 2: Running with Docker

```bash
# 1. Build the production Docker image
docker build -t wavecast-app .

# 2. Spin up the container
docker run -p 8501:8501 wavecast-app

# 3. Open http://localhost:8501 in your browser!
```

---

## 📊 Weather Input Parameters

| Feature | Metric | Description |
| :--- | :--- | :--- |
| **Wave Height (WVHT)** | Meters ($m$) | Signficant swell level from the previous timestamp |
| **Wind Speed (WSPD)** | Meters/Second ($m/s$) | Atmospheric wind speed above ocean surface |
| **Air Pressure (BARO)** | Hectopascals ($hPa$) | Barometric pressure at sea-level |
| **Sea Temp (WTMP)** | Celsius (°C) | Water surface temperature |

---

## 🛠️ Data Buoy Downloader & Model Retraining

You can easily refresh the datasets and retrain the CNN-LSTM sequence model at any time:

```bash
# Download the latest NOAA data files
python download_data.py

# Clean data and combine parameters
python step2_dataset.py

# Perform feature engineering, normalizations, and scale pickles creation
python step3_preprocessing.py

# Train the CNN-LSTM neural net and output training plots
python wave_prediction.py
```

---

## 👨‍💻 Author Info

**Mohammed Shemeer**  
*B.Tech in Artificial Intelligence & Machine Learning*  
Dhanalakshmi Srinivasan University  

*   **LinkedIn**: [mohammed-shemeer-aiml](https://linkedin.com/in/mohammed-shemeer-aiml)
*   **GitHub**: [@muhammedshemeer](https://github.com/muhammedshemeer)

---

## 📄 Licensing

Licensed under the [MIT License](LICENSE). Feel free to modify, build upon, or distribute as desired.

***

<div align="center">
⭐ If you loved this deep-ocean design, give the repository a star!
</div>
