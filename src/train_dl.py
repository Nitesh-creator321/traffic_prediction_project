# src/train_dl.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import matplotlib.pyplot as plt

# === Load the dataset ===
df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")

# === Data Cleaning & Feature Engineering (same as main.py) ===
df = df.drop_duplicates().copy()
df['date_time'] = pd.to_datetime(df['date_time'])
df['temp'] = df['temp'] - 273.15
df['rain_1h'] = df['rain_1h'].fillna(0)
df['snow_1h'] = df['snow_1h'].fillna(0)
df['clouds_all'] = df['clouds_all'].fillna(0)
df['hour'] = df['date_time'].dt.hour
df['weekday'] = df['date_time'].dt.weekday
df['month'] = df['date_time'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Keep only useful columns
df = df[['temp', 'rain_1h', 'snow_1h', 'clouds_all',
         'hour_sin', 'hour_cos', 'weekday', 'month', 'traffic_volume']]

# === Normalize for LSTM ===
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['traffic_volume']])

# Save scaler for later use
joblib.dump(scaler, "scaler_dl.pkl")

# === Create sequences for LSTM ===
def create_sequences(dataset, time_steps=10):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:i + time_steps, 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(scaled_data, time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === Build LSTM Model ===
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(time_steps, 1)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# === Train LSTM ===
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# === Evaluate LSTM ===
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_inv, y_pred_inv)
mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n✅ LSTM Model Results:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# === Save model and metrics ===
model.save("best_dl_model.keras")

dl_results = pd.DataFrame([{
    "Model": "LSTM",
    "MAE": mae,
    "MSE": mse,
    "R2": r2
}])
dl_results.to_csv("dl_results.csv", index=False)
print("\n📊 Saved Deep Learning results to dl_results.csv")
