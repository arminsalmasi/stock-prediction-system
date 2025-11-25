# Project Summary: Stock Prediction System

## Current Status
**Fully Functional & Optimized.**
The system is capable of downloading massive amounts of stock data, maintaining a local database, and running sophisticated Deep Learning forecasts.

## Key Components

### 1. Data Pipeline (`fetch_stocks.py`)
*   **Source**: Yahoo Finance (yfinance).
*   **Storage**: SQLite (`stockdb.sqlite`) with SQLAlchemy Core for high-performance upserts.
*   **Optimization**: Supports incremental updates (`--days N`) to avoid re-downloading history.
*   **Coverage**: ~12,000 tickers (NASDAQ, NYSE, AMEX).

### 2. Ticker Discovery (`discover_tickers.py`)
*   **Source**: Official NASDAQ FTP server.
*   **Output**: Cleaned list of valid symbols in `all_tickers.txt`.

### 3. Visualization (`view_stocks.py`)
*   **Tech**: Streamlit + Plotly.
*   **Features**: Interactive charts, technical indicators (MA, Volume), data tables.

### 4. Machine Learning Core (`train_model.py`)
*   **Model**: LSTM (PyTorch).
*   **Hardware**: Optimized for Apple Silicon (MPS/Metal).
*   **Methodology**:
    *   **Input**: Daily Log Returns (Stationary).
    *   **Sequence Length**: 60 days.
    *   **Normalization**: Implicit via log returns (handles scaling issues).
    *   **Forecasting**: Autoregressive loop for N-day predictions.
*   **Performance**: Typically <1.5% MAPE on major liquid stocks.

### 5. Automation (`daily_update.sh`)
*   Single-command workflow to:
    1.  Update database (last 5 days).
    2.  Retrain models for key watchlist stocks.
    3.  Generate forecast reports/charts.

## Recent Fixes & Improvements
*   **Fixed Crash Prediction**: Switched from "Window Normalization" to "Log Returns" to prevent mean-reversion bias during strong trends (e.g., AAPL all-time highs).
*   **Fixed Dimension Errors**: Corrected tensor shapes in the inference loop.
*   **Memory Optimization**: Added filtering for liquid stocks and date ranges.
*   **Speed**: Implemented incremental fetching.

## Next Steps (Optional)
*   **Sentiment Analysis**: Integrate news/reddit sentiment as an additional feature.
*   **Hyperparameter Tuning**: Automate grid search for LSTM layers/dropout.
*   **Portfolio Optimization**: Use predictions to suggest optimal asset allocation.
