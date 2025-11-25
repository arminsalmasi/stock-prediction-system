# Stock Prediction & Analysis System

A comprehensive toolset for fetching historical stock data, visualizing market trends, and training Deep Learning (LSTM) models for price prediction. Optimized for Apple Silicon (MPS).

## 🚀 Quick Start

### 1. Daily Routine (Update & Forecast)
Run the automated script to fetch the latest data and generate forecasts for your watchlist:
```bash
./daily_update.sh
```
This will:
1. Download the last 5 days of data for ALL tickers.
2. Train models for AAPL, NVDA, MSFT, GOOGL, TSLA, AMZN, META.
3. Save forecast charts (e.g., `AAPL_forecast.png`) in the current directory.

---

## 🛠️ Tools & Scripts

### 1. Data Collection
*   **`discover_tickers.py`**: Downloads official ticker lists from NASDAQ/NYSE (saves to `all_tickers.txt`).
    ```bash
    python3 discover_tickers.py
    ```
*   **`fetch_stocks.py`**: Downloads historical data (Daily, Weekly, Monthly) for tickers and stores it in `stockdb.sqlite`.
    *   **Initial Setup** (Download everything):
        ```bash
        python3 fetch_stocks.py
        ```
    *   **Daily Update** (Fast):
        ```bash
        python3 fetch_stocks.py --days 5
        ```

### 2. Visualization
*   **`view_stocks.py`**: Interactive dashboard to explore data.
    ```bash
    streamlit run view_stocks.py
    ```
    *   Features: Candlestick charts, Moving Averages, Volume Analysis, Raw Data Inspector.

### 3. AI Prediction (Deep Learning)
*   **`train_model.py`**: Trains an LSTM Neural Network on a specific stock to predict future prices.
    *   **Method**: Uses **Daily Log Returns** to handle trends and all-time highs robustly.
    *   **Usage**:
        ```bash
        python3 train_model.py --ticker NVDA --epochs 50 --forecast-days 7
        ```
    *   **Output**:
        *   `NVDA_prediction.png`: Test set performance (Actual vs Predicted).
        *   `NVDA_forecast.png`: 7-day future price projection.
        *   Console output with specific price targets.

---

## 📂 Project Structure

*   `stockdb.sqlite`: Main database (SQLite) containing all historical data.
*   `all_tickers.txt`: List of ~12,000 valid stock tickers.
*   `daily_update.sh`: Automation script for daily maintenance.
*   `venv/`: Python virtual environment.

## ⚙️ Requirements
*   Python 3.9+
*   PyTorch (MPS enabled for Mac)
*   Pandas, NumPy, Scikit-Learn
*   Streamlit, Plotly, YFinance

## 🧠 Model Details
The prediction model is a **Long Short-Term Memory (LSTM)** network designed for time-series forecasting.
*   **Input**: Sequence of last 60 daily log returns.
*   **Architecture**: 2 LSTM layers (64 units) + Dropout + Linear Head.
*   **Training**: Adam Optimizer, MSE Loss.
*   **Inference**: Autoregressive forecasting (feeds predictions back as input for multi-step forecasts).
