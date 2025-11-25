#!/usr/bin/env python3
"""
Discover valid stock tickers by downloading official lists from NASDAQ FTP.
This includes NASDAQ, NYSE, and AMEX listed stocks.
"""

import ftplib
import os
import sys
import pandas as pd
from io import StringIO

NASDAQ_FTP_HOST = 'ftp.nasdaqtrader.com'
NASDAQ_DIR = 'SymbolDirectory'
FILES = ['nasdaqlisted.txt', 'otherlisted.txt']
OUTPUT_FILE = 'all_tickers.txt'

def download_tickers():
    all_tickers = set()
    
    try:
        print(f"🔌 Connecting to {NASDAQ_FTP_HOST}...")
        ftp = ftplib.FTP(NASDAQ_FTP_HOST)
        ftp.login()
        ftp.cwd(NASDAQ_DIR)
        
        for filename in FILES:
            print(f"⬇️  Downloading {filename}...")
            
            # Download file to memory
            r = StringIO()
            def callback(data):
                r.write(data.decode('utf-8'))
            
            ftp.retrbinary(f'RETR {filename}', lambda data: callback(data))
            r.seek(0)
            
            # Parse file
            # These files are pipe-delimited (|)
            # The last line is usually a file creation timestamp which we should ignore
            content = r.getvalue()
            lines = content.strip().splitlines()
            
            # Remove the last line if it looks like a timestamp (e.g. "File Creation Time: ...")
            if lines and "File Creation Time" in lines[-1]:
                lines.pop()
                
            data = StringIO("\n".join(lines))
            df = pd.read_csv(data, sep='|')
            
            # Extract symbols
            # NASDAQ file has 'Symbol', Other listed has 'ACT Symbol'
            col = 'Symbol' if 'Symbol' in df.columns else 'ACT Symbol'
            
            if col in df.columns:
                symbols = df[col].dropna().unique()
                print(f"   Found {len(symbols)} tickers in {filename}")
                all_tickers.update(symbols)
            
        ftp.quit()
        
    except Exception as e:
        print(f"❌ Error downloading from FTP: {e}")
        return []

    # Clean tickers
    # NASDAQ lists often use test tickers or special formats
    cleaned_tickers = []
    for t in all_tickers:
        t = str(t).strip()
        if not t: continue
        
        # Convert NASDAQ special chars to Yahoo format if needed
        # e.g. BRK.B -> BRK-B
        # NASDAQ uses '$' or '.' often for classes
        t = t.replace('$', '-').replace('.', '-')
        
        cleaned_tickers.append(t)
        
    return sorted(list(set(cleaned_tickers)))

def main():
    print("🚀 Starting Ticker Discovery (Source: NASDAQ FTP)")
    tickers = download_tickers()
    
    if tickers:
        print(f"\n✅ Successfully collected {len(tickers)} unique tickers.")
        
        with open(OUTPUT_FILE, 'w') as f:
            for t in tickers:
                f.write(f"{t}\n")
                
        print(f"💾 Saved to {OUTPUT_FILE}")
        print(f"💡 Usage: python fetch_stocks.py --from-file {OUTPUT_FILE}")
    else:
        print("\n⚠️  No tickers found. You may need to try the brute-force method.")

if __name__ == "__main__":
    main()
