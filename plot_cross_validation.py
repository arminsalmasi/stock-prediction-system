#!/usr/bin/env python3
"""
Generate a cross plot of Predicted vs Actual prices from experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def plot_cross_validation(csv_file="experiment_results_detailed.csv"):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Error: {csv_file} not found.")
        return

    # Extract Actual and Predicted values
    # We'll use the 'last_actual_price' and 'last_pred_price' columns
    # which represent the forecast target (whether it's day 1 or day 5)
    actuals = df['last_actual_price']
    preds = df['last_pred_price']
    
    # Calculate R-squared
    r2 = r2_score(actuals, preds)

    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(actuals, preds, alpha=0.6, color='blue', label='Stocks')
    
    # Ideal line (y=x)
    max_val = max(actuals.max(), preds.max())
    min_val = min(actuals.min(), preds.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (Perfect Prediction)')
    
    plt.title(f'Predicted vs Actual Price (R² = {r2:.4f})')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale might be useful if prices vary wildly, but let's stick to linear first
    # unless the range is huge.
    
    output_file = "cross_plot_predicted_vs_actual.png"
    plt.savefig(output_file)
    print(f"✅ Cross plot saved to {output_file}")

if __name__ == "__main__":
    plot_cross_validation()
