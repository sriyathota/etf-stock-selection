"""
HoldingsCSVReader class for reading and processing ETF holdings CSV files
"""

import pandas as pd
import os
from typing import Dict, List, Optional


class HoldingsCSVReader:
    """Reads and processes ETF holdings from CSV files"""
    
    def __init__(self, holdings_dir='src/holdings_data'):
        """
        Initialize the CSV reader
        
        Args:
            holdings_dir (str): Directory containing the holdings CSV files
        """
        self.holdings_dir = holdings_dir
        self.holdings_data = {}
        
    def read_etf_holdings(self, etf_symbol: str) -> Optional[pd.DataFrame]:
        """
        Read holdings for a specific ETF
        
        Args:
            etf_symbol (str): ETF symbol (e.g., 'SPY', 'VTI')
            
        Returns:
            pd.DataFrame: Holdings data or None if file not found
        """
        file_path = os.path.join(self.holdings_dir, f"{etf_symbol}-holdings.csv")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
            
        try:
            df = pd.read_csv(file_path)
            print(f"✅ {etf_symbol}: Loaded {len(df)} holdings from {file_path}")
            return df
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return None
    
    def read_all_holdings(self, etf_symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Read holdings for multiple ETFs
        
        Args:
            etf_symbols (list): List of ETF symbols
            
        Returns:
            dict: Dictionary mapping ETF symbols to their holdings DataFrames
        """
        holdings_dict = {}
        
        for symbol in etf_symbols:
            df = self.read_etf_holdings(symbol)
            if df is not None:
                holdings_dict[symbol] = df
                
        return holdings_dict
    
    def get_holdings_summary(self, holdings_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a summary of holdings across all ETFs
        
        Args:
            holdings_dict (dict): Dictionary of ETF holdings DataFrames
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        summary_data = []
        
        for etf_symbol, df in holdings_dict.items():
            total_holdings = len(df)
            top_holding = df.iloc[0] if len(df) > 0 else None
            
            summary_data.append({
                'ETF': etf_symbol,
                'Total_Holdings': total_holdings,
                'Top_Holding_Symbol': top_holding['Symbol'] if top_holding is not None else 'N/A',
                'Top_Holding_Weight': top_holding['Portfolio Weight'] if top_holding is not None else 'N/A',
                'Top_Holding_Description': top_holding['Description'] if top_holding is not None else 'N/A'
            })
        
        return pd.DataFrame(summary_data)
    
    def get_unique_stocks(self, holdings_dict: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Get list of unique stock symbols across all ETFs
        
        Args:
            holdings_dict (dict): Dictionary of ETF holdings DataFrames
            
        Returns:
            list: List of unique stock symbols
        """
        all_symbols = set()
        
        for df in holdings_dict.values():
            all_symbols.update(df['Symbol'].tolist())
            
        return sorted(list(all_symbols))
    
    def get_stock_etf_mapping(self, holdings_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Create mapping of stock symbols to ETFs that hold them
        
        Args:
            holdings_dict (dict): Dictionary of ETF holdings DataFrames
            
        Returns:
            dict: Mapping of stock symbols to list of ETFs
        """
        stock_to_etfs = {}
        
        for etf_symbol, df in holdings_dict.items():
            for _, row in df.iterrows():
                stock_symbol = row['Symbol']
                if stock_symbol not in stock_to_etfs:
                    stock_to_etfs[stock_symbol] = []
                stock_to_etfs[stock_symbol].append(etf_symbol)
        
        return stock_to_etfs
