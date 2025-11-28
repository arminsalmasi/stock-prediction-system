#!/usr/bin/env python3
"""
Experiment: Random 100 Stocks Validation
1. Selects 100 random stocks.
2. Trains LSTM on all data EXCEPT last 5 days.
3. Forecasts the last 5 days.
4. Calculates MAPE.
"""

import argparse
import sqlite3
import time
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Database configuration
DATABASE_FILE = "stockdb.sqlite"

# -----------------------------------------------------------------------------
# 1. Data Loading & Helpers
# -----------------------------------------------------------------------------

def get_all_tickers():
    """Fetch all distinct symbols from the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM stock_eod WHERE interval='1d'")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers

def load_data(ticker):
    """Load data from SQLite database (Last 2000 days)."""
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
        return None
        
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

def create_log_return_dataset(prices, seq_length):
    """
    Create sequences of log returns.
    """
    prices = np.array(prices)
    # log_ret_t = ln(Price_t / Price_{t-1})
    log_returns = np.log(prices[1:] / prices[:-1])
    
    X, y = [], []
    for i in range(len(log_returns) - seq_length):
        window = log_returns[i:(i + seq_length)]
        target = log_returns[i + seq_length]
        X.append(window)
        y.append(target)
        
    return np.array(X), np.array(y), log_returns

# -----------------------------------------------------------------------------
# 2. Model Definition
# -----------------------------------------------------------------------------

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(-1) 
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# -----------------------------------------------------------------------------
# 3. Experiment Logic
# -----------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt

# ... (imports remain the same, adding os and matplotlib)

# -----------------------------------------------------------------------------
# 3. Experiment Logic
# -----------------------------------------------------------------------------

def plot_forecast(ticker, dates, actuals, predictions, save_dir, days):
    """Plot forecast vs actuals."""
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actuals, label='Actual Price', marker='o', color='black')
    plt.plot(dates, predictions, label='Predicted Price', marker='x', linestyle='--', color='blue')
    plt.title(f'{ticker} Validation Forecast ({days} Days)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"{ticker}_validation.png"))
    plt.close()

def run_experiment(args):
    device = get_device()
    print(f"🚀 Device: {device}")
    
    # Create plots directory
    plots_dir = "experiment_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    all_tickers = get_all_tickers()
    print(f"📊 Found {len(all_tickers)} total tickers in DB.")
    
    # Select random tickers
    if len(all_tickers) < args.count:
        print(f"⚠️  Requested {args.count} tickers but only {len(all_tickers)} available. Using all.")
        selected_tickers = all_tickers
    else:
        selected_tickers = random.sample(all_tickers, args.count)
        
    print(f"🧪 Selected {len(selected_tickers)} tickers for experiment.")
    print(f"📅 Forecast Horizon: {args.days} days")
    
    results = []
    
    # Progress bar
    pbar = tqdm(selected_tickers, unit="ticker")
    
    for ticker in pbar:
        pbar.set_description(f"Processing {ticker}")
        
        try:
            # 1. Load Data
            df = load_data(ticker)
            if df is None or len(df) < 100: # Need enough data
                continue
                
            prices = df['close'].values
            dates = df['date'].values
            
            # 2. Split Data (Hold out last N days)
            validation_days = args.days
            if len(prices) <= validation_days + 60: # Ensure enough training data
                 continue

            train_prices = prices[:-validation_days]
            
            actual_validation_prices = prices[-validation_days:]
            validation_dates = dates[-validation_days:]
            
            # 3. Prepare Training Dataset
            SEQ_LENGTH = 60
            X, y, all_log_returns = create_log_return_dataset(train_prices, SEQ_LENGTH)
            
            if len(X) < 10: # Not enough data after sequence creation
                continue
            
            dataset = StockDataset(X, y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # 4. Train Model
            model = StockLSTM(input_size=1, hidden_size=64, num_layers=2).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            epochs = 20 # Reduced epochs for speed in experiment
            
            for epoch in range(epochs):
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
            
            # 5. Forecast Validation Days
            model.eval()
            
            # Start with the last window of the TRAINING data
            current_returns = all_log_returns[-SEQ_LENGTH:].tolist()
            current_price = train_prices[-1]
            
            pred_prices = []
            
            for i in range(validation_days):
                window_arr = np.array(current_returns[-SEQ_LENGTH:])
                tensor_in = torch.tensor(window_arr, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_log_ret = model(tensor_in).item()
                
                next_price = current_price * np.exp(pred_log_ret)
                
                current_returns.append(pred_log_ret)
                current_price = next_price
                pred_prices.append(next_price)
            
            pred_prices = np.array(pred_prices)
            
            # 6. Calculate Metrics & Store Detailed Results
            mape = np.mean(np.abs((actual_validation_prices - pred_prices) / actual_validation_prices)) * 100
            
            row = {
                'ticker': ticker,
                'mape': mape,
                'last_actual_price': actual_validation_prices[-1],
                'last_pred_price': pred_prices[-1]
            }
            
            # Add daily details
            for i in range(validation_days):
                row[f'date_day_{i+1}'] = validation_dates[i]
                row[f'actual_day_{i+1}'] = actual_validation_prices[i]
                row[f'pred_day_{i+1}'] = pred_prices[i]
                
            results.append(row)
            
            # 7. Plot
            plot_forecast(ticker, validation_dates, actual_validation_prices, pred_prices, plots_dir, args.days)
            
        except Exception as e:
            # print(f"Error processing {ticker}: {e}")
            continue
            
    # 8. Summary
    if not results:
        print("❌ No results generated.")
        return

    df_results = pd.DataFrame(results)
    mean_mape = df_results['mape'].mean()
    median_mape = df_results['mape'].median()
    std_mape = df_results['mape'].std()
    
    print(f"\n{'='*40}")
    print(f"📊 EXPERIMENT RESULTS ({len(results)} stocks, {args.days} Days)")
    print(f"{'='*40}")
    print(f"   Mean MAPE:   {mean_mape:.2f}%")
    print(f"   Median MAPE: {median_mape:.2f}%")
    print(f"   Std Dev:     {std_mape:.2f}%")
    print(f"{'='*40}")
    
    # Save detailed results
    output_file = "experiment_results_detailed.csv"
    df_results.to_csv(output_file, index=False)
    print(f"💾 Detailed results saved to {output_file}")
    print(f"📈 Plots saved to {plots_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100, help='Number of stocks to test')
    parser.add_argument('--days', type=int, default=5, help='Number of days to forecast (validation)')
    args = parser.parse_args()
    
    run_experiment(args)
