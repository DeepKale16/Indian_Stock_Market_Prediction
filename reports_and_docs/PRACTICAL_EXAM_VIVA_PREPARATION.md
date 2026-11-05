# PRACTICAL EXAM VIVA VOCE PREPARATION GUIDE
## “INDIAN STOCK MARKET PREDICTION”
**Candidate:** Deep Dadasaheb Kale (Exam No: 234100452)  
**Guide:** Dr. Sarika Zaware | **Department:** Computer Engineering, AISSMS IOIT Pune  

---

### TOP 25 EXPECTED EXAMINER QUESTIONS & MODEL HIGH-SCORING ANSWERS

#### 1. What is the core problem statement of your project?
> **Answer:** Stock market forecasting is challenging because prices are influenced by non-linear economic factors, market sentiment, and volatility. Traditional statistical models (ARIMA) and standard ML techniques fail to capture long-term sequential dependencies. Most existing systems only look at single closing prices. Our project overcomes this by using a **Bidirectional LSTM** deep neural network combined with multiple technical indicators (MA20, MA50, RSI, MACD) to forecast the **NIFTY 50** index.

#### 2. What dataset did you use and where did it come from?
> **Answer:** We used 5 years of daily historical data for the **NIFTY 50** index (`^NSEI`) collected from the Yahoo Finance API. The dataset includes Open, High, Low, Close, and Volume (OHLCV).

#### 3. Why did you choose NIFTY 50 instead of a single company stock?
> **Answer:** The NIFTY 50 represents the weighted average of the 50 largest and most actively traded Indian companies listed on the National Stock Exchange (NSE). It serves as the primary benchmark for the Indian economy (aligning with **SDG 8: Decent Work and Economic Growth**), providing higher liquidity, lower individual stock manipulation risk, and representative market trends.

#### 4. What are the technical indicators used, and what does each measure?
> **Answer:**
> - **MA20 (20-day Simple Moving Average):** Smooths short-term fluctuations to indicate monthly trend direction.
> - **MA50 (50-day Simple Moving Average):** Indicates medium-term trend direction and support/resistance levels.
> - **RSI (Relative Strength Index - 14 Periods):** Momentum oscillator between 0 and 100. RSI > 70 indicates an overbought condition (potential pullback); RSI < 30 indicates an oversold condition (potential recovery).
> - **MACD (Moving Average Convergence Divergence):** Difference between 12-day EMA and 26-day EMA, plotted with a 9-day Signal line to spot momentum reversals and crossovers.

#### 5. Why use a Bidirectional LSTM (Bi-LSTM) instead of a standard unidirectional LSTM?
> **Answer:** Standard LSTM only processes sequential time-series forward ($t_1 \to t_2 \to \dots \to t_n$). In financial markets, pattern recognition benefits from analyzing context both forward and backward. A **Bidirectional LSTM** trains two parallel LSTM layers—one forward and one backward—and concatenates their outputs ($H_t = [\vec{h}_t \mathbin{\Vert} \overleftarrow{h}_t]$). This allows the network to capture both past price action and backward-looking contextual cycles.

#### 6. What is the lookback time window, and why was 60 days chosen?
> **Answer:** We used a **60-day sliding window**. In the Indian stock market, there are approximately 20 to 22 trading days per month. A 60-day window covers roughly one full quarter (3 months) of market activity, which is the standard quarterly reporting and review cycle for Indian listed companies.

#### 7. How did you split your dataset, and why didn't you use random train_test_split?
> **Answer:** We used a **chronological 80:20 split** (80% training, 20% testing). In time-series forecasting, random shuffling causes severe **data leakage** (future data leaking into the past). We must train on past historical data and test on subsequent future unseen data to reflect real-world forecasting.

#### 8. Why did you use MinMaxScaler instead of StandardScaler?
> **Answer:** LSTM recurrent cells use activation functions like **sigmoid** ($[0, 1]$ in forget, input, and output gates) and **tanh** ($[-1, 1]$ in candidate cell state). `MinMaxScaler(feature_range=(0, 1))` scales all features into $[0, 1]$, perfectly aligning with gate activations and preventing exploding or vanishing gradients during backpropagation through time (BPTT).

#### 9. How did you prevent overfitting in your neural network?
> **Answer:**
> 1. **Dropout Layers ($p=0.2$):** After each Bidirectional LSTM layer, randomly disabling 20% of neuron outputs prevents feature co-adaptation.
> 2. **Early Stopping:** Monitoring validation loss during training with patience of 10 epochs.
> 3. **Train/Validation Split:** 10% validation split during training.

#### 10. Explain the quantitative evaluation results of your model.
> **Answer:** On the normalized test set (as documented in Chapter 5.2):
> - **RMSE = 0.0327:** Low standard deviation of prediction errors.
> - **MAE = 0.0277:** Low absolute deviation from true prices.
> - **MAPE = 16.38%:** Moderate relative error acceptable given high market volatility.
> - **$R^2$ Score = 0.9297 (~0.93):** The model explains ~93% of the variance in NIFTY 50 prices.

#### 11. What is the $R^2$ Score (Coefficient of Determination) formula?
> **Answer:**
> $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$
> An $R^2$ of 1.0 means perfect prediction; 0.93 means 93% of variance is explained by the model.

#### 12. Why does the prediction line look slightly smoother than the actual price line?
> **Answer:** Financial time-series data contains high-frequency stochastic noise (intra-day rumors, block deals, emotional trading). The deep learning model minimizes Mean Squared Error (MSE), learning the conditional expectation $\mathbb{E}[Y \mid X]$, which filters out noise while faithfully tracking the underlying trend.

#### 13. What is the architecture of your neural network?
> **Answer:**
> - Layer 1: `Input(shape=(60, 5))`
> - Layer 2: `Bidirectional(LSTM(64, return_sequences=True))` $\to$ Output shape `(None, 60, 128)`
> - Layer 3: `Dropout(0.2)`
> - Layer 4: `Bidirectional(LSTM(64, return_sequences=False))` $\to$ Output shape `(None, 128)`
> - Layer 5: `Dropout(0.2)`
> - Layer 6: `Dense(32, activation='relu')`
> - Layer 7: `Dense(1, activation='linear')` $\to$ Predicts next day's price

#### 14. What optimizer and loss function did you use?
> **Answer:** We used the **Adam optimizer** (Adaptive Moment Estimation) because it computes adaptive learning rates for each parameter using first and second moments of gradients. The loss function is **Mean Squared Error (MSE)**.

#### 15. How does your system perform multi-step future forecasting?
> **Answer:** Through autoregressive recursive forecasting: the model takes the last 60 known days to predict Day $t+1$. This predicted price is appended to the window while the oldest day is dropped. The updated window then predicts Day $t+2$, and this process repeats for 7 to 30 days.

#### 16. What is the role of Flask and React in your project?
> **Answer:**
> - **Flask (Backend API):** Serves REST API endpoints, loads the saved `.keras` model, performs data preprocessing and normalization on request, and returns predicted prices.
> - **React (Frontend UI):** Provides an interactive dashboard for users to view candlestick charts, check stock news, trigger predictions, and view future trend trajectories.

#### 17. How does this project support UN Sustainable Development Goal 8 (SDG 8)?
> **Answer:** SDG 8 promotes Decent Work and Economic Growth. By democratizing access to reliable stock market forecasting and algorithmic trend analysis, small retail investors and businesses can make data-driven financial decisions and manage capital risk effectively.

#### 18. What are the limitations of your project?
> **Answer:**
> 1. It relies primarily on technical and quantitative price data and does not yet integrate unstructured qualitative data like breaking news sentiment or geopolitical events.
> 2. Currently focused on the NIFTY 50 index rather than individual equities or options/futures derivatives.

#### 19. What is the future scope of your project?
> **Answer:**
> 1. Incorporate **FinBERT** or NLP sentiment analysis on financial news and social media sentiment.
> 2. Implement **Transformer/Attention-based** architectures (such as Temporal Fusion Transformers).
> 3. Expand from NIFTY 50 to individual sectoral indices (Bank NIFTY, NIFTY IT) and individual stock portfolios.

#### 20. How is this project run in Jupyter Notebook during the exam?
> **Answer:** The entire workflow is encapsulated in `Indian_Stock_Market_Prediction.ipynb`. We can step through each section—from dataset loading and technical indicator visualization to model architecture, test evaluation metrics, actual vs predicted plots, and future forecasting—or run the notebook end-to-end.
