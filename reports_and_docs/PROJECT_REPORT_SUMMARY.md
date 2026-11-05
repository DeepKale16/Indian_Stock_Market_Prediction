# A PROJECT REPORT SUMMARY
## “INDIAN STOCK MARKET PREDICTION”
**AI-Powered Indian Stock Market Prediction System Using Bidirectional LSTM**  
**Sustainable Development Goal (SDG 8): Decent Work and Economic Growth**  

- **Institute:** AISSMS’s Institute of Information Technology, Pune  
- **Department:** Computer Engineering  
- **Academic Year:** 2026–2027  
- **Student Name:** Deep Dadasaheb Kale  
- **Exam No:** 234100452  
- **Guide:** Dr. Sarika Zaware  
- **HOD:** Dr. K. S. Wagh  

---

### 1. Abstract Summary
The project aims to develop an intelligent platform that analyzes historical stock market data and predicts future price movements using deep learning techniques. The system focuses on forecasting the performance of the **NIFTY 50** index using historical data obtained from Yahoo Finance (`^NSEI`).

Stock market time-series data is notoriously non-linear, dynamic, and volatile. Traditional statistical approaches (e.g., ARIMA) and basic ML models (e.g., Linear Regression, SVM) fail to model sequential dependencies and non-linear momentum shifts. Moreover, systems that rely only on closing prices experience lower accuracy.

To address these challenges, this project implemented a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network with Dropout layers to capture temporal dependencies in both forward and backward directions. It also integrates key technical indicators:
- **Moving Averages (MA20 and MA50)**
- **Relative Strength Index (RSI - 14 Days)**
- **Moving Average Convergence Divergence (MACD & Signal Line)**

---

### 2. Quantitative Results & Evaluation Metrics (Chapter 5)
The model was tested using historical stock market data divided into an 80:20 training-testing ratio using a 60-day historical sliding window:

| Metric | Normalized Value (Report Benchmark) | Actual INR Scale (₹) | Theoretical Interpretation |
| :--- | :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **0.0327** | ~₹494.61 | Measures standard deviation of residuals; penalizes larger errors heavily. |
| **Mean Absolute Error (MAE)** | **0.0277** | ~₹387.44 | Measures the average magnitude of prediction deviations without direction. |
| **Mean Absolute Percentage Error (MAPE)**| **16.38%** | ~1.62% | Moderate percentage error reflecting acceptable variance in noisy financial markets. |
| **Coefficient of Determination ($R^2$ Score)** | **0.9297 (~0.93)**| **0.9297** | Demonstrates that the model explains approximately 93% of the variance in NIFTY 50 index prices. |

---

### 3. Step-by-Step System Execution Pipeline
1. **Data Collection:** Historical NIFTY 50 OHLCV data acquired from Yahoo Finance API.
2. **Preprocessing:** Handling missing values, cleaning, and formatting timestamps.
3. **Feature Engineering:** Calculating MA20, MA50, RSI(14), and MACD oscillators.
4. **Normalization:** Scaling features to $[0, 1]$ using `MinMaxScaler` to stabilize neural network weights.
5. **Sequence Creation:** Formulating 60-day sliding lookback windows to forecast day $t+1$.
6. **Model Building:** Designing Bidirectional LSTM layers with Dropout ($p=0.2$) and Dense projection.
7. **Model Training:** Optimizing with Adam and Mean Squared Error (MSE) loss with Early Stopping.
8. **Evaluation:** Assessing RMSE, MAE, MAPE, $R^2$, and plotting actual vs. predicted curves.
9. **Deployment:** Serving predictions through a Flask API backend and interactive frontend dashboard.
