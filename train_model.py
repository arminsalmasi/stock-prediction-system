#!/usr/bin/env python3
"""
Train a Single-Ticker Deep Learning model (LSTM) for stock price prediction.
Uses DAILY LOG RETURNS to handle trends and non-stationarity.

Method:
- Input: Sequence of Log Returns: ln(P_t / P_{t-1})
- Output: Next Day's Log Return
- Reconstruction: P_next = P_prev * exp(pred_log_return)

Usage:
    python train_model.py --ticker AAPL --epochs 50 --forecast-days 7
"""

import argparse
import sqlite3
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Database configuration
DATABASE_FILE = "stockdb.sqlite"

# -----------------------------------------------------------------------------
# 1. Data Loading
# -----------------------------------------------------------------------------

def load_data(ticker):
    """Load data from SQLite database (Last 2000 days)."""
    print(f"⏳ Loading data for {ticker}...")
    conn = sqlite3.connect(DATABASE_FILE)
    query = """
        SELECT * FROM (
            SELECT date, close 
            FROM stock_eod 
            WHERE symbol = ? AND interval = '1d'
            ORDER BY date DESC
            LIMIT 2000
        ) ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    
    if df.empty:
        print(f"❌ No data found for {ticker}")
        sys.exit(1)
        
    df['date'] = pd.to_datetime(df['date'])
    return df

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------------------------------------------------------
# 2. Log Return Logic
# -----------------------------------------------------------------------------

def create_log_return_dataset(prices, seq_length):
    """
    Create sequences of log returns.
    """
    # Calculate Log Returns
    # log_ret_t = ln(Price_t / Price_{t-1})
    # We lose the first data point
    prices = np.array(prices)
    log_returns = np.log(prices[1:] / prices[:-1])
    
    X, y = [], []
    for i in range(len(log_returns) - seq_length):
        window = log_returns[i:(i + seq_length)]
        target = log_returns[i + seq_length]
        X.append(window)
        y.append(target)
        
    return np.array(X), np.array(y), log_returns

# -----------------------------------------------------------------------------
# 3. Model Definition
# -----------------------------------------------------------------------------

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len) -> need (batch, seq_len, features)
        x = x.unsqueeze(-1) 
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------------------------------------------------------
# 4. Training Pipeline
# -----------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_model(args):
    device = get_device()
    print(f"🚀 Device: {device}")
    
    # 1. Load Data
    df = load_data(args.ticker)
    prices = df['close'].values
    print(f"✅ Data loaded: {len(prices)} records.")
    
    # 2. Create Dataset
    SEQ_LENGTH = 60
    X, y, all_log_returns = create_log_return_dataset(prices, SEQ_LENGTH)
    
    # Split Train/Test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Model
    model = StockLSTM(input_size=1, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Train
    print(f"\n🧠 Training {args.ticker} model for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.6f}")
            
    print(f"✅ Done in {time.time() - start_time:.2f}s")
    
    # 5. Evaluate
    model.eval()
    pred_returns = []
    actual_returns = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_returns.extend(outputs.cpu().numpy())
            actual_returns.extend(targets.numpy())
            
    pred_returns = np.array(pred_returns).flatten()
    actual_returns = np.array(actual_returns).flatten()
    
    # Reconstruct Prices for Test Set
    # We need the price just before the test set starts to reconstruct the series
    # Test set starts at index `train_size` of X.
    # X[train_size] corresponds to log_returns[train_size : train_size+SEQ_LENGTH]
    # The target y[train_size] is log_returns[train_size+SEQ_LENGTH]
    # This corresponds to the price change from index (train_size+SEQ_LENGTH) to (train_size+SEQ_LENGTH+1)
    
    # Let's align indices carefully.
    # The test set targets correspond to returns starting from index (train_size + SEQ_LENGTH) of `all_log_returns`
    # The price corresponding to index `i` in log_returns is `prices[i+1]`
    # So the first predicted price corresponds to `prices[train_size + SEQ_LENGTH + 1]`
    # The base price for reconstruction is `prices[train_size + SEQ_LENGTH]`
    
    test_start_idx = train_size + SEQ_LENGTH
    base_prices = prices[test_start_idx : test_start_idx + len(pred_returns)]
    
    # P_next = P_curr * exp(ret)
    pred_prices = base_prices * np.exp(pred_returns)
    actual_prices = base_prices * np.exp(actual_returns)
    
    # Metrics
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    print(f"\n📊 Model Performance:")
    print(f"   MAPE: {mape:.2f}%")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Price', color='black', alpha=0.7)
    plt.plot(pred_prices, label='Predicted Price', color='blue', alpha=0.7)
    plt.title(f'{args.ticker} Price Prediction (Log Returns)')
    plt.xlabel('Days (Test Set)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{args.ticker}_prediction.png")
    print(f"📈 Chart saved to {args.ticker}_prediction.png")
    
    # 6. Predict Next N Days
    print(f"\n🔮 FORECAST FOR NEXT {args.forecast_days} DAYS:")
    print(f"   Current Price: ${prices[-1]:.2f}")
    print(f"   {'-'*40}")
    
    # Start with the last known window of returns
    current_returns = all_log_returns[-SEQ_LENGTH:].tolist()
    current_price = prices[-1]
    
    future_dates = []
    future_prices = []
    
    # Generate future dates
    last_date = df['date'].iloc[-1]
    current_date = last_date
    
    model.eval()
    
    for i in range(args.forecast_days):
        # 1. Prepare input
        window_arr = np.array(current_returns[-SEQ_LENGTH:])
        tensor_in = torch.tensor(window_arr, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 2. Predict
        with torch.no_grad():
            pred_log_ret = model(tensor_in).item()
            
        # 3. Update Price
        next_price = current_price * np.exp(pred_log_ret)
        
        # 4. Update state
        current_returns.append(pred_log_ret)
        current_price = next_price
        future_prices.append(next_price)
        
        # Add date
        current_date = current_date + pd.Timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += pd.Timedelta(days=1)
        future_dates.append(current_date)
        
        change = next_price - prices[-1]
        pct = (change / prices[-1]) * 100
        daily_change_pct = (np.exp(pred_log_ret) - 1) * 100
        
        print(f"   Day +{i+1} ({current_date.strftime('%Y-%m-%d')}): ${next_price:.2f}  (Daily: {daily_change_pct:+.2f}%)")
        
    # Plot Forecast
    plt.figure(figsize=(10, 6))
    history_dates = df['date'].iloc[-60:]
    history_prices = prices[-60:]
    
    plt.plot(history_dates, history_prices, label='History', color='black')
    plt.plot(future_dates, future_prices, label='Forecast', color='red', linestyle='--', marker='o')
    plt.title(f'{args.ticker} {args.forecast_days}-Day Forecast')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{args.ticker}_forecast.png")
    print(f"\n📈 Forecast chart saved to {args.ticker}_forecast.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--forecast-days', type=int, default=1)
    args = parser.parse_args()
    
    train_model(args)
