import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras
from keras.models import Sequential
from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Load Data
csv_path = os.path.join(base_dir, "Dataset", "NIFTY50_historical_data.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows.")

features = ['Close', 'MA20', 'MA50', 'RSI', 'MACD']
data = df[features].values

# 2. Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(df[['Close']].values)

# 3. Create Sequences (60 days lookback)
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i])
    y.append(scaled_data[i, 0])  # Predicting 'Close' price (index 0)

X, y = np.array(X), np.array(y)
print(f"Total sequences created: {X.shape}, labels: {y.shape}")

# 4. Train-Test Split (80:20 ratio)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# 5. Build Bidirectional LSTM Architecture
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 6. Train Model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 7. Evaluate on Test Set
y_pred_scaled = model.predict(X_test).flatten()

# Normalized metrics
rmse_norm = float(np.sqrt(mean_squared_error(y_test, y_pred_scaled)))
mae_norm = float(mean_absolute_error(y_test, y_pred_scaled))
# Calculate MAPE avoiding division by near-zero:
mape_norm = float(np.mean(np.abs((y_test - y_pred_scaled) / np.maximum(y_test, 1e-5))) * 100)
r2_norm = float(r2_score(y_test, y_pred_scaled))

# Inverse transform to INR
y_test_inr = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inr = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

rmse_inr = float(np.sqrt(mean_squared_error(y_test_inr, y_pred_inr)))
mae_inr = float(mean_absolute_error(y_test_inr, y_pred_inr))
mape_inr = float(np.mean(np.abs((y_test_inr - y_pred_inr) / y_test_inr)) * 100)
r2_inr = float(r2_score(y_test_inr, y_pred_inr))

print("\n--- Model Evaluation (Normalized Scale) ---")
print(f"RMSE:     {rmse_norm:.4f}")
print(f"MAE:      {mae_norm:.4f}")
print(f"MAPE:     {mape_norm:.2f}%")
print(f"R2 Score: {r2_norm:.4f}")

print("\n--- Model Evaluation (Actual INR Scale) ---")
print(f"RMSE:     ₹{rmse_inr:.2f}")
print(f"MAE:      ₹{mae_inr:.2f}")
print(f"MAPE:     {mape_inr:.2f}%")
print(f"R2 Score: {r2_inr:.4f}")

# 8. Save Model and Scalers
model_save_path = os.path.join(base_dir, "models", "nifty50_bilstm_model.keras")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

scaler_save_path = os.path.join(base_dir, "models", "scaler.pkl")
with open(scaler_save_path, 'wb') as f:
    pickle.dump({'feature_scaler': scaler, 'target_scaler': target_scaler}, f)
print(f"Scalers saved to {scaler_save_path}")

metrics_dict = {
    "normalized": {
        "rmse": round(rmse_norm, 4),
        "mae": round(mae_norm, 4),
        "mape": round(mape_norm, 2),
        "r2_score": round(r2_norm, 4)
    },
    "inr_scale": {
        "rmse": round(rmse_inr, 2),
        "mae": round(mae_inr, 2),
        "mape": round(mape_inr, 2),
        "r2_score": round(r2_inr, 4)
    },
    "report_benchmark": {
        "rmse": 0.0327,
        "mae": 0.0277,
        "mape": 16.38,
        "r2_score": 0.9297
    }
}

with open(os.path.join(base_dir, "models", "metrics.json"), "w") as f:
    json.dump(metrics_dict, f, indent=4)

# 9. Generate and Save Plots
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)', color='#2563eb', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='#dc2626', linestyle='--', linewidth=2)
plt.title('Bidirectional LSTM Training & Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "static", "loss_curve.png"), dpi=200)
plt.close()

# Actual vs Predicted Plot
test_dates = df['Date'].iloc[time_step + train_size:].values
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_inr, label='Actual NIFTY 50 Close (INR)', color='#1e293b', linewidth=2)
plt.plot(test_dates, y_pred_inr, label='Predicted NIFTY 50 Close (Bi-LSTM)', color='#10b981', linestyle='--', linewidth=2)
plt.title('NIFTY 50: Actual vs Predicted Closing Prices (Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('NIFTY 50 Index (₹)', fontsize=12)
plt.xticks(test_dates[::max(1, len(test_dates)//8)], rotation=30)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "static", "actual_vs_predicted.png"), dpi=200)
plt.close()

print("Plots saved successfully in static directory.")
