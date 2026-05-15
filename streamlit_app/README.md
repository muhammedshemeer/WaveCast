# WaveCast | Streamlit Ocean Wave Height Prediction Dashboard

A premium, immersive, ocean-themed Streamlit dashboard for CNN-LSTM wave-height prediction, fully self-contained in Python.

---

## 🛠️ Project Stack & Libraries
- **Streamlit** (v1.36.0+)
- **Pandas & NumPy**
- **Plotly** (v5.20.0+) for high-end data visualization
- **TensorFlow** (v2.15.0+) for CNN-LSTM deep learning inference
- **Scikit-Learn & Joblib** for MinMaxScaler scaling

---

## 🚀 How to Run Locally

1. **Activate the Virtual Environment**:
   ```bash
   venv\Scripts\activate
   ```
2. **Install Dependencies** (if needed):
   ```bash
   pip install -r streamlit_app/requirements.txt
   ```
3. **Launch the Dashboard**:
   ```bash
   streamlit run streamlit_app/app.py
   ```

---

## 📂 Directory Structure
```
streamlit_app/
├── app.py                  # Core Streamlit app containing custom UI and cached inference
├── requirements.txt        # Streamlit-specific Python requirements
├── models/
│   ├── wave_prediction_model.h5  # CNN-LSTM Keras H5/Keras model file
│   ├── x_scaler.pkl              # Pickled feature scaler
│   └── y_scaler.pkl              # Pickled target prediction scaler
└── README.md               # App documentation
```
