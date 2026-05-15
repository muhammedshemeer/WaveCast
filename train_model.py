import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. DATA PREPARATION (REALISTIC GENERATION)
# ==========================================
print("Generating realistic NOAA meteorological buoy data...")

# Create 30 days of hourly ocean readings (720 rows)
np.random.seed(42)
hours = np.arange(720)
timestamps = pd.date_range(start="2026-04-15", periods=720, freq="h")

# Simulate weather fluctuations using overlapping sine waves and random noise
wind_speed = 7.0 + 4.5 * np.sin(hours / 24.0) + np.random.normal(0, 1.8, 720)
wind_speed = np.clip(wind_speed, 0.5, 28.0) # Bound to logical wind metrics

air_pressure = 1013.0 - 8.0 * np.sin(hours / 36.0) + np.random.normal(0, 2.5, 720)
sea_temp = 12.0 + 3.0 * np.sin(hours / 360.0) + np.random.normal(0, 0.2, 720)

# Calculate wave height based on physical parameters with noise
wave_height = 0.4 + 0.08 * (wind_speed ** 1.3) + 0.03 * (1020.0 - air_pressure) + np.random.normal(0, 0.15, 720)
wave_height = np.clip(wave_height, 0.2, 6.5)

# Build a Pandas DataFrame
df = pd.DataFrame({
    "Timestamp": timestamps,
    "WaveHeight": wave_height,
    "WindSpeed": wind_speed,
    "AirPressure": air_pressure,
    "SeaTemp": sea_temp
})

# Ensure target folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save raw dataset to csv for tracking
csv_path = "data/raw_buoy_data.csv"
df.to_csv(csv_path, index=False)
print(f"Raw dataset saved to {csv_path} ({len(df)} records)")

# ==========================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ==========================================
print("Engineering time-series lag features...")

# Create lag feature: Previous hour's wave height is a strong predictor of current wave height
df["PrevWaveHeight"] = df["WaveHeight"].shift(1)

# Drop first row which contains NaN due to shifting
df_model = df.dropna().reset_index(drop=True)

# Define feature columns and target variable
features = ["WindSpeed", "AirPressure", "SeaTemp", "PrevWaveHeight"]
X = df_model[features]
y = df_model["WaveHeight"]

# ==========================================
# 3. MODEL TRAINING & VALIDATION SPLIT
# ==========================================
print("Splitting dataset (80% Train / 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor Model...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. PERFORMANCE EVALUATION METRICS
# ==========================================
print("Evaluating model performance...")
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Model Test Metrics ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} meters")
print(f"Coefficient of Determination (R2): {r2:.4f}")
print("-----------------------------\n")

# Display feature importances
print("Feature Importances (Weight %):")
for feat, imp in zip(features, model.feature_importances_):
    print(f"   * {feat}: {imp * 100:.2f}%")

# ==========================================
# 5. MODEL SERIALIZATION (SAVING)
# ==========================================
model_path = "models/wave_model.joblib"
joblib.dump(model, model_path)
print(f"\nModel successfully serialized and saved to {model_path}")
print("ML training workflow complete!")
