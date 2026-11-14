"""
ETFAnalyzer - Main orchestrator class that combines all components
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

from holdings_downloader import ETFHoldingsDownloader
from holdings_csv_reader import HoldingsCSVReader
from yfinance_data_fetcher import YFinanceDataFetcher


class ETFAnalyzer:
    """Main orchestrator class that combines all ETF analysis components"""
    
    def __init__(self, 
                 holdings_dir='src/holdings_data',
                 data_period='2y',
                 max_stocks_per_etf=None,
                 headless=True):
        """
        Initialize the ETF analyzer
        
        Args:
            holdings_dir (str): Directory for holdings CSV files
            data_period (str): Period for yfinance data (default: '1y')
            max_stocks_per_etf (int): Maximum stocks to analyze per ETF
            headless (bool): Whether to run browser in headless mode
        """
        self.holdings_dir = holdings_dir
        self.max_stocks_per_etf = max_stocks_per_etf
        self.headless = headless
        
        # Initialize components
        self.downloader = ETFHoldingsDownloader(output_dir=holdings_dir, headless=headless)
        self.csv_reader = HoldingsCSVReader(holdings_dir=holdings_dir)
        self.yf_fetcher = YFinanceDataFetcher(period=data_period, interval='1d')
        
        # Data storage
        self.holdings_data = {}
        self.stock_data = {}
        
    def download_etf_holdings(self, etf_symbols: List[str], force_download: bool = False) -> Dict[str, str]:
        """
        Download holdings for specified ETFs
        
        Args:
            etf_symbols (list): List of ETF symbols to download
            force_download (bool): Whether to re-download existing files
            
        Returns:
            dict: Results of download process
        """
        print("📥 Downloading ETF Holdings")
        print("=" * 50)
        
        if force_download:
            print("🔄 Force download enabled - will re-download all ETFs")
        else:
            print("📁 Checking existing files first...")
        
        results = {}
        for symbol in etf_symbols:
            file_path = os.path.join(self.holdings_dir, f"{symbol}-holdings.csv")
            
            if not force_download and os.path.exists(file_path):
                print(f"✅ {symbol}: File already exists at {file_path}")
                results[symbol] = file_path
            else:
                print(f"📥 {symbol}: Downloading fresh data...")
                if self.downloader._get_etf_from_schwab(symbol):
                    results[symbol] = file_path
                    print(f"✅ {symbol}: Download completed")
                else:
                    print(f"❌ {symbol}: Download failed")
                    results[symbol] = None
        
        return results
    
    def load_holdings_data(self, etf_symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load holdings data from CSV files
        
        Args:
            etf_symbols (list): List of ETF symbols to load
            
        Returns:
            dict: Dictionary mapping ETF symbols to holdings DataFrames
        """
        print("\n📊 Loading Holdings Data")
        print("=" * 50)
        
        self.holdings_data = self.csv_reader.read_all_holdings(etf_symbols)
        
        # Print summary and display sample dataframes
        for symbol, df in self.holdings_data.items():
            print(f"✅ {symbol}: {len(df)} holdings loaded")
            
            # Display holdings dataframe
            print(f"\n📊 {symbol} Holdings DataFrame ({df.shape[0]} rows × {df.shape[1]} columns):")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Top 5 holdings:")
            print(df.head().to_string())
            print(f"   Last 5 holdings:")
            print(df.tail().to_string())
        
        return self.holdings_data
    
    def fetch_stock_data(self, etf_symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch yfinance data for all holdings
        
        Args:
            etf_symbols (list): List of ETF symbols to process
            
        Returns:
            dict: Dictionary mapping ETF symbols to their stock data
        """
        print("\n📈 Fetching Stock Data")
        print("=" * 50)
        
        if not self.holdings_data:
            print("❌ No holdings data loaded. Run load_holdings_data() first.")
            return {}
        
        self.stock_data = {}
        
        for etf_symbol in etf_symbols:
            if etf_symbol not in self.holdings_data:
                print(f"❌ {etf_symbol}: No holdings data available")
                continue
            
            print(f"\n📊 Processing {etf_symbol}...")
            holdings_df = self.holdings_data[etf_symbol]
            
            # Limit to top holdings
            if self.max_stocks_per_etf is not None and len(holdings_df) > self.max_stocks_per_etf:
                holdings_df = holdings_df.head(self.max_stocks_per_etf)
                print(f"   Limited to top {self.max_stocks_per_etf} holdings")
            
            symbols = holdings_df['Symbol'].tolist()
            print(f"   Fetching data for {len(symbols)} stocks...")
            
            stock_data = self.yf_fetcher.fetch_multiple_stocks(symbols, delay=0.1)
            self.stock_data[etf_symbol] = stock_data
            
            print(f"✅ {etf_symbol}: {len(stock_data)} stocks with data")
        
        return self.stock_data
    
    def flatten_for_pca(self, etf_symbol: str, include_static: bool = False) -> pd.DataFrame:
        """
        Flatten stock data into PCA-ready DataFrame
        
        Args:
            etf_symbol (str): ETF symbol to flatten
            include_static (bool): Whether to include static features as columns
            
        Returns:
            pd.DataFrame: All stocks' daily data stacked
                         Shape: (N_stocks × N_days rows × features columns)
        """
        if etf_symbol not in self.stock_data:
            print(f"❌ {etf_symbol}: No stock data available")
            return pd.DataFrame()
        
        all_stocks = []
        for stock_symbol, stock_info in self.stock_data[etf_symbol].items():
            daily_df = stock_info['daily_data'].copy()
            daily_df['Date'] = daily_df.index  # Preserve dates as column
            daily_df['Stock_Symbol'] = stock_symbol
            
            if include_static:
                # Add static features as repeated columns
                for key, value in stock_info['static_features'].items():
                    daily_df[f'static_{key}'] = value
            
            all_stocks.append(daily_df)
        
        return pd.concat(all_stocks, ignore_index=True)
    
    def run_complete_analysis(self, etf_symbols: List[str], 
                            force_download: bool = False,
                            max_stocks_per_etf: Optional[int] = None) -> Dict[str, Dict]:
        """
        Run complete ETF analysis pipeline
        
        Args:
            etf_symbols (list): List of ETF symbols to analyze
            force_download (bool): Whether to re-download holdings
            max_stocks_per_etf (int): Override max stocks per ETF
            
        Returns:
            dict: {ETF: {Stock: {daily_data: DataFrame, static_features: dict}}}
        """
        print("🚀 Starting Complete ETF Analysis")
        print("=" * 60)
        
        if max_stocks_per_etf is not None:
            self.max_stocks_per_etf = max_stocks_per_etf
        
        # Step 1: Download holdings
        self.download_etf_holdings(etf_symbols, force_download)
        
        # Step 2: Load holdings data
        self.load_holdings_data(etf_symbols)
        
        # Step 3: Fetch stock data
        stock_data = self.fetch_stock_data(etf_symbols)
        
        print(f"\n✅ Analysis Complete!")
        return stock_data
    


def main():
    """Main function to run the ETF analyzer"""
    analyzer = ETFAnalyzer(
        holdings_dir='src/holdings_data',
        data_period='2y',  # Can be '1y', '2y', '5y', etc.
        max_stocks_per_etf=50,
        headless=True
    )
    
    etf_symbols = ['SPY', 'VTI', 'QQQ', 'VOO']
    
    # Get stock data
    stock_data = analyzer.run_complete_analysis(
        etf_symbols=etf_symbols,
        force_download=False
    )
    
    # Example: Flatten for PCA
    spy_pca_data = analyzer.flatten_for_pca('SPY', include_static=False)
    print(f"\nSPY PCA-ready data shape: {spy_pca_data.shape}")
    
    return stock_data


if __name__ == "__main__":
    main()
