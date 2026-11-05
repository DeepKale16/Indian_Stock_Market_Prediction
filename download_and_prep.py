import os
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

print("Fetching NIFTY 50 (^NSEI) data from Yahoo Finance...")

url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=5y&interval=1d"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers, timeout=15)
response.raise_for_status()
data = response.json()

result = data['chart']['result'][0]
timestamps = result['timestamp']
indicators = result['indicators']['quote'][0]

df = pd.DataFrame({
    'Date': pd.to_datetime(timestamps, unit='s').strftime('%Y-%m-%d'),
    'Open': indicators.get('open'),
    'High': indicators.get('high'),
    'Low': indicators.get('low'),
    'Close': indicators.get('close'),
    'Volume': indicators.get('volume')
})

# Drop null values if any
df = df.dropna().reset_index(drop=True)
print(f"Downloaded {len(df)} records. Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

# Technical Indicator Calculations
# 1. Moving Averages: MA20 and MA50
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# 2. RSI (14 periods)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# 3. MACD (12-period EMA - 26-period EMA)
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Drop rows with NaN resulting from rolling windows (first 49 rows)
clean_df = df.dropna().reset_index(drop=True)
print(f"Cleaned dataset shape: {clean_df.shape}")
print(clean_df.head())

# Save to CSV
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "Dataset", "NIFTY50_historical_data.csv")
clean_df.to_csv(csv_path, index=False)
print(f"Dataset successfully saved to {csv_path}")
