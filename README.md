# 🌊 Underwater Wave Prediction System & Dashboard (WaveCast)

This project utilizes a CNN-LSTM deep learning model to predict underwater wave heights using meteorological data (Wind Speed, Pressure, and Temperature) from NOAA buoys, served via a beautiful, self-contained **Streamlit** dashboard.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![R2 Score](https://img.shields.io/badge/R²%20Score-0.96-brightgreen?style=for-the-badge)

**A production-grade, immersive deep learning system that predicts underwater wave heights using a hybrid CNN-LSTM neural network, served via a gorgeous self-contained Streamlit ocean dashboard.**

[Report Bug](#) · [LinkedIn](https://linkedin.com/in/mohammed-shemeer-aiml)

</div>

---

## 📈 Graphical Representation 

| Actual vs Predicted | Model Learning Curve |
|-----------|-----------------|
| ![Dashboard](prediction_plot.png) | ![Result](learning_curve.png) |

---

## 🎯 What This Project Does

Most wave prediction tools require expensive hardware and complex setups. This system provides **real-time wave height predictions** through a simple, beautiful, glassmorphic Streamlit web interface — useful for:

- 🚢 Ship navigation safety
- 🏄 Coastal activity planning  
- ⚡ Offshore energy operations
- 🔬 Oceanographic research

---

## 🧠 Model Architecture

```
Input (10 time steps × 4 features)
        ↓
   Conv1D (64 filters, kernel=2, ReLU)
        ↓
   MaxPooling1D (pool_size=2)
        ↓
   LSTM (50 units, tanh)
        ↓
   Dense (1 unit)
        ↓
Output: Predicted Wave Height (meters)
```

### Why CNN-LSTM?
- **Conv1D layers** capture local spatial/sensor patterns across the sliding time window.
- **LSTM layers** capture long-term temporal dependencies.
- Together they achieve **R² = 0.96** on test data.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| R² Score | **0.96** |
| RMSE | Low |
| MAE | Low |
| Training Data | NOAA NDBC Station 46059 (2018–2023) |

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras |
| Interactive Dashboard | Streamlit |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Containerization | Docker |
| Data Visualizations | Plotly |
| Data Source | NOAA National Data Buoy Center |

---

## 📁 Project Structure

```
Wave_prediction/
├── streamlit_app/
│   ├── app.py              # Main Streamlit app containing UI & logic
│   ├── requirements.txt    # Streamlit requirements
│   ├── models/             # App model folder
│   │   ├── wave_prediction_model.h5  # Compiled CNN-LSTM model
│   │   ├── x_scaler.pkl              # Pickled feature scaler
│   │   └── y_scaler.pkl              # Pickled target scaler
│   └── README.md           # App-specific documentation
├── models/
│   ├── wave_prediction_model.h5    # Root models (legacy references)
│   ├── feature_scaler.pkl
│   └── target_scaler.pkl
├── data/
│   ├── raw/                 # Raw NOAA buoy data
│   └── processed/           # Preprocessed tensors
├── download_data.py         # NOAA data downloader
├── step2_dataset.py         # Dataset preparation
├── step3_preprocessing.py   # Feature engineering & scaling
├── wave_prediction.py       # Model training script
├── Dockerfile               # Container configuration
├── requirements.txt         # Root requirements
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/muhammedshemeer/Wave_prediction.git
cd Wave_prediction

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Streamlit Dashboard
streamlit run streamlit_app/app.py

# 5. Open http://localhost:8501 in your browser
```

### Option 2 — Run with Docker

```bash
# 1. Build the image
docker build -t wave-prediction-app .

# 2. Run the container
docker run -p 8501:8501 wave-prediction-app

# 3. Visit http://localhost:8501 in your browser
```

---

## 📈 Input Features

| Feature | Unit | Description |
|---------|------|-------------|
| Wave Height | meters | Previous recorded significant wave height |
| Wind Speed | m/s | Wind speed over ocean surface |
| Air Pressure | hPa | Atmospheric pressure |
| Temperature | °C | Sea surface temperature |

> The model uses a **sliding window of 10 time steps** to capture temporal patterns before making a prediction.

---

## 🔄 How to Retrain

```bash
# Step 1: Download fresh NOAA data
python download_data.py

# Step 2: Build dataset
python step2_dataset.py

# Step 3: Preprocess & create sequences
python step3_preprocessing.py

# Step 4: Train model
python wave_prediction.py
```

---

## 👨‍💻 Author

**Mohammed Shemeer**  
B.Tech AI & ML | Dhanalakshmi Srinivasan University  
🔗 [LinkedIn](https://linkedin.com/in/mohammed-shemeer-aiml) · [GitHub](https://github.com/muhammedshemeer)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
⭐ Star this repo if you found it useful!
</div>
