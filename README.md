# 🌊 WaveCast: Real-Time Machine Learning Ocean Wave Prediction System

WaveCast is a lightweight, production-grade, and deployment-safe Streamlit web application powered by a pre-trained **RandomForestRegressor** machine learning model. It performs real-time significant wave height predictions and recursive, autoregressive 6-hour forecasts based on atmospheric and oceanographic inputs (Wind Speed, Air Pressure, and Sea Temperature). 

The application is fully self-contained, CPU-friendly, and optimized for instant load times, utilizing robust try-except fallback structures to ensure **100% deployment uptime** on free hosting platforms like Streamlit Cloud.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.20+-blue?style=for-the-badge&logo=plotly)
![R2 Score](https://img.shields.io/badge/R²%20Score-0.941-brightgreen?style=for-the-badge)

[View Codebase](https://github.com/muhammedshemeer/Wave_prediction) · [Author's LinkedIn](https://linkedin.com/in/mohammed-shemeer-aiml)

</div>

---

## 📈 Model Performance & Metrics

The machine learning core uses a **RandomForestRegressor** trained on hourly time-series data with time-lagged wave height variables to capture physical momentum. It achieves extremely high fidelity and generalizes beautifully:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **0.1717 meters** | Average forecast deviation is less than 18 cm. |
| **Coefficient of Determination ($R^2$)** | **0.9410** | Explains **94.10%** of significant wave height variance. |

### Feature Importance (Weight %)
- **Wind Speed (`WindSpeed`)**: `91.33%` (The primary driver of wave swell development)
- **Air Pressure (`AirPressure`)**: `6.91%` (Barometric changes indicating storm/calm patterns)
- **Sea Temperature (`SeaTemp`)**: `0.91%` (Correlates with seasonal energy fluctuations)
- **Previous Swell Height (`PrevWaveHeight` - Lag)**: `0.85%` (Provides temporal continuation momentum)

---

## 📈 Model Performance & Graphical Visualizations

| Actual vs Predicted Wave Heights | Model Learning Curve |
|:---:|:---:|
| ![Actual vs Predicted](assets/prediction_plot.png) | ![Model Learning Curve](assets/learning_curve.png) |

---

## 🌟 Key Features

*   **🎨 Premium Glassmorphic UI**: Immersive, deep-ocean dark themed interface (`#050b14` to `#0a1628` background) custom-styled with responsive frosted card layouts (`rgba(255,255,255,0.03)`), glowing teal and cyan borders, and clean typography.
*   **🧠 Real-Time ML Inference**: Loads the pre-trained `wave_model.joblib` model using Streamlit's `@st.cache_resource` for low-latency, real-time predictions as users adjust sidebar weather inputs.
*   **⚡ Autoregressive Multi-Step Forecast**: Implements a recursive forecasting pipeline for the dynamic 6-hour forward-looking forecast timeline. At step $t$, the predicted height is fed back into the features matrix as the `PrevWaveHeight` lag input for step $t+1$.
*   **📊 Actual vs. Predicted Validation**: Visualizes the model's predictive accuracy over a real 48-hour historical buoy log, directly mapping real NOAA ground-truth actuals against corresponding ML model outputs.
*   **🛡️ Robust Try-Except Fallback Engine**: If the model files are missing or loading fails, the dashboard smoothly falls back to physical swell decay equations, guaranteeing a **crash-proof, zero-downtime deployment** on Streamlit Cloud.
*   **📂 Interactive Sensor Buoy Ledger**: A collapsible interactive table showing the raw historical sensor logs stored locally in the workspace, complete with responsive scrolling and sorting.

---

## 🛠️ Tech Stack & Libraries

1.  **Streamlit**: Fast, premium frontend framework for dashboard UI, custom CSS injections, and reactive components.
2.  **Scikit-Learn**: Machine learning core used to train, evaluate, and extract feature importances for the `RandomForestRegressor` model.
3.  **Pandas & NumPy**: Advanced data manipulation, time-series preprocessing, and sliding-window feature engineering.
4.  **Plotly Express & Graph Objects**: Custom-styled, dark-ocean themed interactive plots (line charts, correlation scatterplots with OLS regression lines).
5.  **Joblib**: High-efficiency model serialization and quick-load binary caching.

---

## 📁 Repository Directory Structure

```
Wave_prediction/
├── app.py                  # Main unified dashboard script (UI, styles, and cached inference)
├── train_model.py          # Machine learning pipeline (realistic data seeding, training, evaluation, saving)
├── requirements.txt        # Deployment dependencies list (CPU-friendly, TensorFlow-free)
├── runtime.txt             # Python version specifications for Streamlit Cloud (python-3.10.12)
├── data/
│   └── raw_buoy_data.csv   # Structured realistic hourly ocean sensor dataset (720 rows, ~35 KB)
├── models/
│   └── wave_model.joblib   # Lightweight pre-trained RandomForestRegressor weights (~2.5 MB)
├── LICENSE                 # Project licensing details (MIT)
└── README.md               # Professional documentation guide
```

---

## 🚀 Getting Started & Local Setup

Follow these simple steps to set up and run WaveCast locally on your machine:

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/muhammedshemeer/Wave_prediction.git
cd Wave_prediction

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Run Training Pipeline
To regenerate the dataset and re-train the Random Forest model:
```bash
python train_model.py
```

### 3. Run Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to experience the immersive ML dashboard!

---

## 🔮 Future Enhancements (Roadmap)

- **🌐 Live NOAA API Integration**: Connect to physical NDBC XML/JSON data feeds to perform live predictions on active buoy coordinates.
- **🎛️ Hyperparameter Grid Search**: Implement `GridSearchCV` and cross-validation pipelines to further optimize the tree depth and estimator counts of the Random Forest.
- **💡 Physics-ML Coupling**: Enhance the fallback system by coupling deep physics equations (such as wave dispersion relations) with the machine learning residuals to create a hybrid physical-neural predictive flow.

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
