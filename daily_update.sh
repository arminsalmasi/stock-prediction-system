#!/bin/bash

# Daily Stock Update & Forecast Script
# Usage: ./daily_update.sh

echo "=================================================="
echo "🚀 STARTING DAILY STOCK UPDATE"
echo "=================================================="
date

# 1. Activate Virtual Environment
source venv/bin/activate

# 2. Update Data (Last 5 days to cover weekends/holidays)
echo ""
echo "📥 Fetching latest data (Last 5 days)..."
python3 fetch_stocks.py --days 5

# 3. Run Forecasts for Watchlist
WATCHLIST=("AAPL" "NVDA" "MSFT" "GOOGL" "TSLA" "AMZN" "META")
EPOCHS=50
FORECAST_DAYS=7

echo ""
echo "🔮 Generating Forecasts for Watchlist..."
echo "   Tickers: ${WATCHLIST[*]}"

for ticker in "${WATCHLIST[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "🧠 Training Model for $ticker..."
    python3 train_model.py --ticker "$ticker" --epochs $EPOCHS --forecast-days $FORECAST_DAYS
done

echo ""
echo "=================================================="
echo "✅ DAILY UPDATE COMPLETE"
echo "=================================================="
echo "Charts saved in current directory."
