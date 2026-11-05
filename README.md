# Indian Stock Market Prediction System (NIFTY 50)
### **B.Tech Computer Engineering Practical Exam Project**
**AISSMS’s Institute of Information Technology, Pune**  
**Candidate:** Deep Dadasaheb Kale (Exam No: 234100452)  
**Guide:** Dr. Sarika Zaware  
**SDG 8:** Decent Work and Economic Growth  

---

## 📌 Project Overview
This repository contains the complete implementation of the **“AI-Powered Indian Stock Market Prediction System”**, exactly aligned with the official project report submitted to AISSMS IOIT Pune.

The system forecasts future closing price movements of the **NIFTY 50** benchmark index using a **Bidirectional Long Short-Term Memory (Bi-LSTM)** deep neural network trained on historical price data along with four key technical indicators:
1. **20-Day Simple Moving Average (MA20)**
2. **50-Day Simple Moving Average (MA50)**
3. **Relative Strength Index (RSI - 14 Days)**
4. **Moving Average Convergence Divergence (MACD & Signal Line)**

---

## 📁 Repository Structure (`E:\Project\Indian_Stock_Market_Prediction`)
```
E:\Project\Indian_Stock_Market_Prediction\
├── Indian_Stock_Market_Prediction.ipynb  <-- MASTER JUPYTER NOTEBOOK (Pre-Executed)
├── app.py                               <-- Interactive Web Dashboard & REST API
├── requirements.txt                     <-- Python Dependencies
├── README.md                            <-- Complete Documentation & Instructions
│
├── Dataset\
│   └── NIFTY50_historical_data.csv      <-- 5-Year Historical Dataset with Indicators
│
├── models\
│   ├── nifty50_bilstm_model.keras       <-- Trained Bidirectional LSTM Deep Learning Model
│   ├── scaler.pkl                       <-- Saved MinMaxScaler Objects (Feature & Target)
│   └── metrics.json                     <-- Benchmark & Model Metrics (RMSE, MAE, MAPE, R2)
│
├── static\
│   ├── actual_vs_predicted.png          <-- Test Set Comparison Chart
│   └── loss_curve.png                   <-- Training vs Validation Loss Curve
│
└── reports_and_docs\
    ├── PROJECT_REPORT_SUMMARY.md        <-- Complete Summary of Project Report
    └── PRACTICAL_EXAM_VIVA_PREPARATION.md<-- Top 20 Examiner Viva Q&A Guide
```

---

## 🚀 How to Run the Jupyter Notebook for the Practical Exam

### Option 1: Using VS Code (Recommended & Fastest)
1. Open **VS Code**.
2. Press `Ctrl + O` or click **File -> Open File...** and select:
   `E:\Project\Indian_Stock_Market_Prediction\Indian_Stock_Market_Prediction.ipynb`
3. Notice that **all outputs, tables, model summaries, and visual plots are already pre-rendered and saved in the file**. You can immediately show it to the external examiner!
4. If asked to run the notebook live, select the Python kernel (`Python 3.12` or `Python 3.14`) in the top right corner and click **Run All**.

### Option 2: Using Jupyter Notebook Browser Interface
1. Open PowerShell or Command Prompt.
2. Change directory to the project folder:
   ```powershell
   cd E:\Project\Indian_Stock_Market_Prediction
   ```
3. Start the Jupyter Notebook server:
   ```powershell
   python -m notebook
   # or
   jupyter notebook Indian_Stock_Market_Prediction.ipynb
   ```
4. The notebook will open in your default web browser at `http://localhost:8888`.

---

## 🌐 How to Run the Web Dashboard & API (Optional Demonstration)
In addition to the Jupyter Notebook, a web application is included to demonstrate the end-to-end deployment described in Chapter 4 of the report:

1. Open PowerShell in `E:\Project\Indian_Stock_Market_Prediction`:
   ```powershell
   cd E:\Project\Indian_Stock_Market_Prediction
   python app.py
   ```
2. Open your web browser and navigate to:
   `http://127.0.0.1:5000`
3. You will see:
   - Live NIFTY 50 benchmark prices.
   - Project metadata (Deep Dadasaheb Kale, Dr. Sarika Zaware, AISSMS IOIT, SDG 8).
   - Evaluation metrics cards matching the project report.
   - Interactive future price forecast table (7, 15, and 30 days ahead).
   - Test set evaluation plots.
   - REST API endpoint: `http://127.0.0.1:5000/api/predict?days=7`

---

## 📊 Summary of Quantitative Results (Chapter 5)
| Metric | Report Benchmark Value | Formula |
| :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | **0.0327** | $\sqrt{\frac{1}{N}\sum(y_i - \hat{y}_i)^2}$ |
| **Mean Absolute Error (MAE)** | **0.0277** | $\frac{1}{N}\sum\|y_i - \hat{y}_i\|$ |
| **Mean Absolute Percentage Error (MAPE)** | **16.38%** | $\frac{100\%}{N}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ |
| **Coefficient of Determination ($R^2$ Score)** | **0.9297 (~0.93)** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ |

---

## 🎯 Viva Voce Quick Cheat Sheet
- **Why Bi-LSTM?** Processes temporal sequences both forward and backward, allowing the model to capture past patterns and contextual reversals.
- **Why 60 days?** In Indian stock trading (~20 trading days/month), 60 days covers roughly one financial quarter (3 months).
- **Why technical indicators?** Single closing prices lack trend and momentum signals. Adding MA20, MA50, RSI, and MACD provides rich market context.
- **Why Dropout (0.2)?** Randomly deactivates 20% of recurrent units during training to prevent overfitting on noisy financial data.
- **Why R² = 0.93?** The model successfully accounts for 93% of the variance in NIFTY 50 price movements.
