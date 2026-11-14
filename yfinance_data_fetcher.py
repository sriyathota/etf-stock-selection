"""
YFinanceDataFetcher class for fetching stock data from yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time


class YFinanceDataFetcher:
    """Fetches stock data from yfinance for ETF holdings"""
    
    def __init__(self, period='1y', interval='1d'):
        """
        Initialize the yfinance data fetcher
        
        Args:
            period (str): Period for historical data (default: '1y' for 1 year)
            interval (str): Data interval (default: '1d' for daily)
        """
        self.period = period
        self.interval = interval
        self.stock_data = {}
        self.failed_symbols = []
        
    def fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch comprehensive data for a single stock
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Dictionary containing daily_data DataFrame and static_features dict
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data for the specified period
            hist = ticker.history(period=self.period, interval=self.interval)
            if hist.empty:
                print(f"❌ {symbol}: No historical data available")
                return None

            hist = hist.copy()
            
            # Get info for static features
            info = ticker.info
            
            # Calculate daily returns
            hist['Daily_Return'] = hist['Close'].pct_change()
            
            # Calculate volatility (rolling 30-day)
            hist['Volatility_30d'] = hist['Daily_Return'].rolling(window=30).std()
            
            # Calculate moving averages
            hist['MA_20'] = hist['Close'].rolling(window=20).mean()
            hist['MA_50'] = hist['Close'].rolling(window=50).mean()
            
            # Calculate RSI (simplified)
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Extract static features
            static_features = {
                'market_cap': info.get('marketCap', np.nan),
                'pe_ratio': info.get('trailingPE', np.nan),
                'beta': info.get('beta', np.nan),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'dividend_yield': info.get('dividendYield', np.nan),
                'profit_margins': info.get('profitMargins', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'return_on_equity': info.get('returnOnEquity', np.nan)
            }
            
            print(f"✅ {symbol}: {len(hist)} days of data, {len([k for k, v in static_features.items() if not pd.isna(v)])} static features")
            
            return {
                'daily_data': hist,
                'static_features': static_features
            }
            
        except Exception as e:
            print(f"❌ {symbol}: Error fetching data - {e}")
            self.failed_symbols.append(symbol)
            return None
    
    def fetch_multiple_stocks(self, symbols: List[str], delay: float = 0.1) -> Dict[str, Dict]:
        """
        Fetch data for multiple stocks with rate limiting
        
        Args:
            symbols (list): List of stock symbols
            delay (float): Delay between requests in seconds
            
        Returns:
            dict: Dictionary mapping symbols to their data
        """
        stock_data = {}
        total = len(symbols)
        
        print(f"📊 Fetching data for {total} stocks...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"   [{i}/{total}] Fetching {symbol}...")
            
            data = self.fetch_stock_data(symbol)
            if data is not None:
                stock_data[symbol] = data
            
            # Rate limiting
            if i < total:  # Don't delay after the last request
                time.sleep(delay)
        
        print(f"✅ Successfully fetched data for {len(stock_data)}/{total} stocks")
        if self.failed_symbols:
            print(f"❌ Failed symbols: {self.failed_symbols}")
        
        return stock_data
    
    
    def get_etf_holdings_data(self, holdings_dict: Dict[str, pd.DataFrame], max_stocks_per_etf: int = 50) -> Dict[str, Dict]:
        """
        Fetch data for all holdings across multiple ETFs
        
        Args:
            holdings_dict (dict): Dictionary of ETF holdings DataFrames
            max_stocks_per_etf (int): Maximum stocks to fetch per ETF (for testing)
            
        Returns:
            dict: Dictionary mapping ETF symbols to their stock data
        """
        etf_stock_data = {}
        
        for etf_symbol, holdings_df in holdings_dict.items():
            print(f"\n📈 Processing {etf_symbol} ({len(holdings_df)} holdings)...")
            
            # Get top holdings (by weight or limit by count)
            if len(holdings_df) > max_stocks_per_etf:
                # Sort by portfolio weight and take top N
                holdings_df = holdings_df.head(max_stocks_per_etf)
                print(f"   Limited to top {max_stocks_per_etf} holdings")
            
            symbols = holdings_df['Symbol'].tolist()
            stock_data = self.fetch_multiple_stocks(symbols)
            
            etf_stock_data[etf_symbol] = stock_data
        
        return etf_stock_data
