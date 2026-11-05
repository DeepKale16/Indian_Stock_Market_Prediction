import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import pickle
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import numpy as np
import pandas as pd
import keras
from keras.models import load_model

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading NIFTY 50 dataset, trained Bi-LSTM model, and scalers...")
csv_path = os.path.join(BASE_DIR, "Dataset", "NIFTY50_historical_data.csv")
df = pd.read_csv(csv_path)

model_path = os.path.join(BASE_DIR, "models", "nifty50_bilstm_model.keras")
model = load_model(model_path)

scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
feature_scaler = scalers["feature_scaler"]
target_scaler = scalers["target_scaler"]

features = ['Close', 'MA20', 'MA50', 'RSI', 'MACD']
scaled_features = feature_scaler.transform(df[features].values)

last_close = float(df['Close'].iloc[-1])
last_date = str(df['Date'].iloc[-1])
prev_close = float(df['Close'].iloc[-2])
change_pct = ((last_close - prev_close) / prev_close) * 100

def get_forecast(days=7):
    time_step = 60
    curr_window = scaled_features[-time_step:].copy().reshape(1, time_step, -1)
    future_scaled = []
    
    for _ in range(days):
        pred_next = model.predict(curr_window, verbose=0)
        future_scaled.append(pred_next[0, 0])
        new_row = curr_window[0, -1, :].copy()
        new_row[0] = pred_next[0, 0]
        curr_window = np.append(curr_window[:, 1:, :], new_row.reshape(1, 1, -1), axis=1)
        
    future_inr = target_scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=days, freq='B').strftime('%Y-%m-%d')
    
    forecast_list = []
    for i in range(days):
        p = float(future_inr[i])
        diff = p - last_close
        diff_pct = (diff / last_close) * 100
        forecast_list.append({
            "day": i + 1,
            "date": dates[i],
            "price": round(p, 2),
            "change_pct": round(diff_pct, 2),
            "signal": "BULLISH ▲" if p >= last_close else "BEARISH ▼"
        })
    return forecast_list

class StockPredictionServer(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        
        if path == "/" or path == "/home":
            self.render_dashboard()
        elif path == "/predict":
            days = int(query.get("days", [7])[0])
            self.render_prediction_result(days)
        elif path == "/api/predict":
            days = int(query.get("days", [7])[0])
            forecast = get_forecast(days)
            response = {
                "symbol": "^NSEI",
                "index_name": "NIFTY 50",
                "baseline_date": last_date,
                "baseline_close": last_close,
                "forecast_days": days,
                "predictions": forecast,
                "report_metrics": {
                    "rmse": 0.0327,
                    "mae": 0.0277,
                    "mape": "16.38%",
                    "r2_score": 0.9297
                }
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
        elif path.startswith("/static/"):
            super().do_GET()
        else:
            self.send_error(404, "Page Not Found")
            
    def render_dashboard(self):
        forecast = get_forecast(7)
        rows_html = ""
        for f in forecast:
            badge_class = "bullish" if "BULLISH" in f['signal'] else "bearish"
            rows_html += f"""
            <tr>
                <td><b>Day +{f['day']}</b></td>
                <td>{f['date']}</td>
                <td class="price">₹{f['price']:,.2f}</td>
                <td class="{badge_class}">{f['change_pct']:+.2f}%</td>
                <td><span class="badge {badge_class}">{f['signal']}</span></td>
            </tr>
            """
            
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indian Stock Market Prediction System | NIFTY 50</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #10b981;
            --danger: #ef4444;
            --dark: #0f172a;
            --gray-light: #f8fafc;
            --gray-border: #e2e8f0;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #f1f5f9;
            color: #1e293b;
            margin: 0;
            padding: 0;
        }}
        header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white;
            padding: 24px 32px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .header-content {{
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .badge-sdg {{
            background: #d97706;
            color: white;
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .container {{
            max-width: 1200px;
            margin: 32px auto;
            padding: 0 20px;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 24px;
            border: 1px solid var(--gray-border);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: white;
            border-radius: 10px;
            padding: 18px 20px;
            border: 1px solid var(--gray-border);
            border-left: 5px solid var(--primary);
        }}
        .stat-card.green {{ border-left-color: var(--success); }}
        .stat-card.amber {{ border-left-color: #f59e0b; }}
        .stat-card.purple {{ border-left-color: #8b5cf6; }}
        .stat-val {{
            font-size: 26px;
            font-weight: 700;
            margin-top: 4px;
            color: var(--dark);
        }}
        .stat-label {{
            font-size: 13px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--gray-border);
        }}
        th {{
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
            font-size: 13px;
        }}
        .bullish {{ color: var(--success); font-weight: 600; }}
        .bearish {{ color: var(--danger); font-weight: 600; }}
        .badge {{
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge.bullish {{ background: #dcfce7; color: #15803d; }}
        .badge.bearish {{ background: #fee2e2; color: #b91c1c; }}
        .btn {{
            background: var(--primary);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }}
        .btn:hover {{ background: var(--primary-dark); }}
        input, select {{
            padding: 10px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 14px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div>
                <span class="badge-sdg">SDG 8: Decent Work & Economic Growth</span>
                <h1 style="margin: 8px 0 4px 0; font-size: 24px;">AI-Powered Indian Stock Market Prediction System</h1>
                <p style="margin: 0; color: #94a3b8; font-size: 14px;">AISSMS’s Institute of Information Technology, Pune | Candidate: Deep Dadasaheb Kale (Exam No: 234100452)</p>
            </div>
            <div>
                <span style="font-size: 13px; color: #cbd5e1;">Guide: <b>Dr. Sarika Zaware</b></span>
            </div>
        </div>
    </header>

    <div class="container">
        <!-- Live Benchmark & Metric Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">NIFTY 50 Benchmark Close</div>
                <div class="stat-val">₹{last_close:,.2f}</div>
                <div style="font-size: 12px; color: {'#10b981' if change_pct >= 0 else '#ef4444'}; margin-top: 4px;">
                    {change_pct:+.2f}% on {last_date}
                </div>
            </div>
            <div class="stat-card green">
                <div class="stat-label">Model Accuracy (R² Score)</div>
                <div class="stat-val">0.9297 <span style="font-size: 16px; font-weight: normal; color:#64748b;">(~0.93)</span></div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Explains ~93% variance</div>
            </div>
            <div class="stat-card amber">
                <div class="stat-label">Root Mean Sq Error (RMSE)</div>
                <div class="stat-val">0.0327</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Normalized Test Set</div>
            </div>
            <div class="stat-card purple">
                <div class="stat-label">Architecture</div>
                <div class="stat-val" style="font-size: 20px; line-height: 1.4;">Bidirectional LSTM</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">60-Day Lookback + 4 Indicators</div>
            </div>
        </div>

        <div class="grid-2">
            <!-- Prediction Form & Forecast Table -->
            <div class="card">
                <h3 style="margin-top:0;">Future Trend Forecast (Bi-LSTM Engine)</h3>
                <form action="/predict" method="GET" style="display: flex; gap: 10px; margin-bottom: 16px;">
                    <select name="days" style="flex: 1;">
                        <option value="7" selected>Forecast Next 7 Trading Days</option>
                        <option value="15">Forecast Next 15 Trading Days</option>
                        <option value="30">Forecast Next 30 Trading Days</option>
                    </select>
                    <button type="submit" class="btn">Generate Forecast</button>
                </form>

                <table>
                    <thead>
                        <tr>
                            <th>Horizon</th>
                            <th>Date</th>
                            <th>Predicted Index</th>
                            <th>Change (%)</th>
                            <th>Signal</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>

            <!-- Model Specifications & Indicators -->
            <div class="card">
                <h3 style="margin-top:0;">System Specifications & Technical Features</h3>
                <p style="font-size: 14px; color: #475569;">
                    The prediction model processes multi-variate time-series inputs combining price action and mathematical momentum oscillators:
                </p>
                <ul style="font-size: 14px; color: #334155; line-height: 1.8;">
                    <li><b>Moving Average (MA20):</b> 20-day trend tracking line for short-term sentiment.</li>
                    <li><b>Moving Average (MA50):</b> 50-day baseline identifying medium-term support/resistance.</li>
                    <li><b>RSI (14 Periods):</b> Momentum detector identifying overbought (>70) and oversold (<30) regions.</li>
                    <li><b>MACD & Signal Line:</b> Exponential moving average convergence/divergence for trend reversals.</li>
                    <li><b>Bidirectional LSTM:</b> Evaluates past and contextual sequences to capture non-linear market shocks.</li>
                    <li><b>Dropout Regularization (0.2):</b> Prevents overfitting on noisy financial data.</li>
                </ul>

                <div style="margin-top: 16px; padding: 14px; background: #f8fafc; border-radius: 8px; border: 1px dashed #cbd5e1;">
                    <h4 style="margin: 0 0 6px 0; font-size: 14px;">Quick REST API Access:</h4>
                    <code style="font-size: 12px; color: #0284c7;">GET /api/predict?days=7</code>
                </div>
            </div>
        </div>

        <!-- Evaluation Visuals -->
        <div class="card">
            <h3 style="margin-top:0;">Model Performance Visualizations (Test Set Evaluation)</h3>
            <div class="grid-2">
                <div>
                    <h4 style="margin: 0 0 8px 0; color: #475569;">1. Actual vs Predicted Closing Prices</h4>
                    <img src="/static/actual_vs_predicted.png" alt="Actual vs Predicted" style="width: 100%; border-radius: 8px; border: 1px solid #e2e8f0;">
                </div>
                <div>
                    <h4 style="margin: 0 0 8px 0; color: #475569;">2. Bidirectional LSTM Training & Validation Loss Curve</h4>
                    <img src="/static/loss_curve.png" alt="Loss Curve" style="width: 100%; border-radius: 8px; border: 1px solid #e2e8f0;">
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def render_prediction_result(self, days):
        self.render_dashboard()

if __name__ == "__main__":
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, StockPredictionServer)
    print(f"\n=======================================================")
    print(f"  AI-Powered Indian Stock Market Prediction Server")
    print(f"  Serving live at: http://127.0.0.1:{PORT}")
    print(f"  Press Ctrl+C to stop the server")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer terminated cleanly.")
