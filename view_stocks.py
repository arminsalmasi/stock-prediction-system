#!/usr/bin/env python3
"""
Interactive Stock Data Viewer with GUI
Usage: streamlit run view_stocks.py
       python view_stocks.py (for command-line mode)
"""

import sqlite3
import pandas as pd
import sys
from datetime import datetime

DATABASE = 'stockdb.sqlite'

# Command-line mode functions
def get_summary():
    """Show database summary."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    print(f"\n{'='*70}")
    print(f"📊 STOCK DATABASE SUMMARY")
    print(f"{'='*70}")
    
    cursor.execute('SELECT COUNT(*) FROM stock_eod')
    total = cursor.fetchone()[0]
    print(f"\n📈 Total Records: {total:,}")
    
    print(f"\n📋 Stocks in Database:\n")
    cursor.execute('''
        SELECT 
            symbol, 
            interval,
            COUNT(*) as records,
            MIN(date) as earliest, 
            MAX(date) as latest
        FROM stock_eod 
        GROUP BY symbol, interval
        ORDER BY symbol, interval
    ''')
    
    print(f"  {'Symbol':<8} {'Interval':<8} {'Records':>8}  {'Date Range':<30}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")
    for row in cursor.fetchall():
        print(f"  {row[0]:<8} {row[1]:<8} {row[2]:>8,}  {row[3]} to {row[4]}")
    
    conn.close()
    print(f"\n{'='*70}\n")


def get_latest_prices():
    """Show latest prices for all stocks."""
    conn = sqlite3.connect(DATABASE)
    
    print(f"💰 LATEST CLOSING PRICES")
    print(f"{'='*70}\n")
    
    cursor = conn.cursor()
    cursor.execute('''
        SELECT symbol, date, close, volume
        FROM stock_eod
        WHERE date = (SELECT MAX(date) FROM stock_eod) AND interval = '1d'
        ORDER BY symbol
    ''')
    
    print(f"  {'Symbol':<8} {'Price':>10}  {'Volume':>15}  {'Date':<12}")
    print(f"  {'-'*8} {'-'*10}  {'-'*15}  {'-'*12}")
    
    for row in cursor.fetchall():
        symbol, date, close, volume = row
        volume_str = f"{volume:,}" if volume else "N/A"
        print(f"  {symbol:<8} ${close:>9.2f}  {volume_str:>15}  {date:<12}")
    
    conn.close()
    print(f"\n{'='*70}\n")


# Check if we're running in Streamlit
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def streamlit_app():
    """Main Streamlit GUI application."""
    
    # Page config
    st.set_page_config(
        page_title="Stock Data Viewer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {font-size:20px !important; font-weight:bold;}
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("📈 Stock Data Viewer")
    st.markdown("---")
    
    # Connect to database
    conn = sqlite3.connect(DATABASE)
    
    # Sidebar - Filters
    st.sidebar.header("🔍 Search & Filter")
    
    # Get available symbols
    symbols_df = pd.read_sql_query("""
        SELECT DISTINCT symbol 
        FROM stock_eod 
        ORDER BY symbol
    """, conn)
    symbols = symbols_df['symbol'].tolist()
    
    # Search box
    search_term = st.sidebar.text_input("🔎 Search Stock Symbol", "", 
                                        help="Type to search for stocks")
    
    # Filter symbols based on search
    if search_term:
        filtered_symbols = [s for s in symbols if search_term.upper() in s.upper()]
    else:
        filtered_symbols = symbols
    
    # Dropdown for stock selection
    selected_symbol = st.sidebar.selectbox(
        "📊 Select Stock",
        options=filtered_symbols if filtered_symbols else ["No stocks found"],
        help="Choose a stock to analyze"
    )
    
    # Time interval selection
    available_intervals = pd.read_sql_query("""
        SELECT DISTINCT interval 
        FROM stock_eod 
        WHERE symbol = ?
        ORDER BY interval
    """, conn, params=(selected_symbol,))
    
    if not available_intervals.empty:
        interval = st.sidebar.selectbox(
            "⏰ Time Resolution",
            options=available_intervals['interval'].tolist(),
            help="Select time interval"
        )
    else:
        interval = '1d'
    
    # Date range filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Date Range")
    
    # Get min/max dates for selected stock
    date_range_query = pd.read_sql_query("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM stock_eod
        WHERE symbol = ? AND interval = ?
    """, conn, params=(selected_symbol, interval))
    
    if not date_range_query.empty and date_range_query['min_date'][0]:
        min_date = pd.to_datetime(date_range_query['min_date'][0]).date()
        max_date = pd.to_datetime(date_range_query['max_date'][0]).date()
        
        date_from = st.sidebar.date_input("From", min_date, min_value=min_date, max_value=max_date)
        date_to = st.sidebar.date_input("To", max_date, min_value=min_date, max_value=max_date)
    else:
        date_from = datetime.now().date()
        date_to = datetime.now().date()
    
    # Main content
    if selected_symbol and selected_symbol != "No stocks found":
        
        # Get stock data
        query = """
            SELECT * FROM stock_eod
            WHERE symbol = ? AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, 
                               params=(selected_symbol, interval, date_from, date_to))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            
            # Metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            latest_price = df.iloc[-1]['close']
            prev_price = df.iloc[-2]['close'] if len(df) > 1 else latest_price
            price_change = latest_price - prev_price
            price_change_pct = (price_change / prev_price * 100) if prev_price else 0
            
            with col1:
                st.metric(
                    label="💵 Latest Price",
                    value=f"${latest_price:.2f}",
                    delta=f"{price_change_pct:+.2f}%"
                )
            
            with col2:
                st.metric(
                    label="📊 Records",
                    value=f"{len(df):,}"
                )
            
            with col3:
                st.metric(
                    label="📈 High",
                    value=f"${df['high'].max():.2f}"
                )
            
            with col4:
                st.metric(
                    label="📉 Low",
                    value=f"${df['low'].min():.2f}"
                )
            
            with col5:
                st.metric(
                    label="📅 Days",
                    value=f"{(df['date'].max() - df['date'].min()).days}"
                )
            
            st.markdown("---")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Volume", "📋 Data Table", "📉 Statistics"])
            
            with tab1:
                st.subheader(f"💹 {selected_symbol} Price Chart ({interval})")
                
                # Candlestick chart
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=('Price', 'Volume')
                )
                
                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df['date'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ),
                    row=1, col=1
                )
                
                # Volume bars
                colors = ['red' if row['open'] > row['close'] else 'green' 
                         for _, row in df.iterrows()]
                
                fig.add_trace(
                    go.Bar(
                        x=df['date'],
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors,
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=700,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Moving averages
                st.subheader("📊 Moving Averages")
                col1, col2 = st.columns(2)
                
                with col1:
                    ma_periods = st.multiselect(
                        "Select MA Periods",
                        [5, 10, 20, 50, 100, 200],
                        default=[20, 50]
                    )
                
                if ma_periods:
                    fig_ma = go.Figure()
                    
                    fig_ma.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['close'],
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    for period in ma_periods:
                        if len(df) >= period:
                            ma = df['close'].rolling(window=period).mean()
                            fig_ma.add_trace(go.Scatter(
                                x=df['date'],
                                y=ma,
                                name=f'MA{period}',
                                line=dict(width=1.5)
                            ))
                    
                    fig_ma.update_layout(
                        height=400,
                        hovermode='x unified',
                        template='plotly_white',
                        xaxis_title="Date",
                        yaxis_title="Price ($)"
                    )
                    
                    st.plotly_chart(fig_ma, use_container_width=True)
            
            with tab2:
                st.subheader(f"📊 {selected_symbol} Volume Analysis")
                
                fig_vol = px.bar(
                    df,
                    x='date',
                    y='volume',
                    title='Trading Volume Over Time',
                    labels={'volume': 'Volume', 'date': 'Date'},
                    color='volume',
                    color_continuous_scale='Blues'
                )
                
                fig_vol.update_layout(height=500, template='plotly_white')
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Volume statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Volume", f"{df['volume'].mean():,.0f}")
                with col2:
                    st.metric("Max Volume", f"{df['volume'].max():,.0f}")
                with col3:
                    st.metric("Min Volume", f"{df['volume'].min():,.0f}")
            
            with tab3:
                st.subheader(f"📋 {selected_symbol} Data Table")
                
                # Format the dataframe
                display_df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                display_df = display_df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Display with formatting
                st.dataframe(
                    display_df.style.format({
                        'Open': '${:.2f}',
                        'High': '${:.2f}',
                        'Low': '${:.2f}',
                        'Close': '${:.2f}',
                        'Volume': '{:,.0f}'
                    }),
                    height=500,
                    use_container_width=True
                )
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_symbol}_{interval}_{date_from}_{date_to}.csv",
                    mime="text/csv"
                )
            
            with tab4:
                st.subheader(f"📉 {selected_symbol} Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Price Statistics")
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                        'Value': [
                            f"${df['close'].mean():.2f}",
                            f"${df['close'].median():.2f}",
                            f"${df['close'].std():.2f}",
                            f"${df['close'].min():.2f}",
                            f"${df['close'].max():.2f}",
                            f"${df['close'].max() - df['close'].min():.2f}"
                        ]
                    })
                    st.table(stats_df)
                
                with col2:
                    st.markdown("### Returns")
                    
                    returns = df['close'].pct_change()
                    
                    returns_df = pd.DataFrame({
                        'Metric': ['Daily Avg Return', 'Total Return', 'Volatility', 'Sharpe Ratio (approx)'],
                        'Value': [
                            f"{returns.mean()*100:.2f}%",
                            f"{((df['close'].iloc[-1] / df['close'].iloc[0]) - 1)*100:.2f}%",
                            f"{returns.std()*100:.2f}%",
                            f"{(returns.mean() / returns.std()):.2f}" if returns.std() > 0 else "N/A"
                        ]
                    })
                    st.table(returns_df)
                
                # Returns distribution
                st.markdown("### Returns Distribution")
                fig_hist = px.histogram(
                    returns.dropna() * 100,
                    nbins=50,
                    title='Daily Returns Distribution (%)',
                    labels={'value': 'Return (%)', 'count': 'Frequency'},
                    template='plotly_white'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        else:
            st.warning(f"No data found for {selected_symbol} with interval {interval}")
    
    else:
        st.info("👈 Please select a stock from the sidebar")
        
        # Show database overview
        st.subheader("📊 Database Overview")
        
        overview_query = """
            SELECT 
                symbol,
                interval,
                COUNT(*) as records,
                MIN(date) as earliest,
                MAX(date) as latest,
                MIN(close) as min_price,
                MAX(close) as max_price
            FROM stock_eod
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """
        overview_df = pd.read_sql_query(overview_query, conn)
        
        if not overview_df.empty:
            st.dataframe(overview_df, use_container_width=True, height=400)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stocks", len(overview_df['symbol'].unique()))
            with col2:
                st.metric("Total Records", f"{overview_df['records'].sum():,}")
            with col3:
                st.metric("Intervals", len(overview_df['interval'].unique()))
    
    conn.close()


# Main entry point
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE and len(sys.argv) == 1:
        # GUI mode
        streamlit_app()
    else:
        # Command-line mode
        if len(sys.argv) > 1:
            if sys.argv[1] == '--summary':
                get_summary()
            elif sys.argv[1] == '--prices':
                get_latest_prices()
            else:
                print("Usage: streamlit run view_stocks.py  (for GUI)")
                print("       python view_stocks.py --summary")
                print("       python view_stocks.py --prices")
        else:
            if not STREAMLIT_AVAILABLE:
                print("⚠️  Streamlit not installed. Install with: pip install streamlit plotly")
                print("\nUsing command-line mode instead...")
                get_summary()
            else:
                print("Run with: streamlit run view_stocks.py")
