"""
MimicETF - PCA-based ETF mimicking analysis

This module implements PCA analysis to find the minimum number of stocks
that best mimic an ETF's behavior using cumulative explained variance.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

from etf_analyzer import ETFAnalyzer


def create_output_dir(etf_symbol):
        # Create output directory for plots
        output_dir = f"src/pca/pca_results/{etf_symbol}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


def fetch_etf_returns(etf_symbol, start):
    """Fetch returns for any ETF from start date"""
    ticker = yf.Ticker(etf_symbol)
    hist = ticker.history(start=start, interval='1d')
    returns = hist['Close'].pct_change().dropna()
    returns.name = etf_symbol
    return returns


class MimicETF:
    """
    Analyzes which stocks best mimic ETF behavior using PCA
    
    Goal: Find minimum number of stocks that capture maximum variance
    to mimic ETF behavior with as few stocks as possible.
    """
    
    def __init__(self, analyzer: ETFAnalyzer, variance_threshold: float = 0.25):
        """
        Initialize the MimicETF analyzer
        
        Args:
            analyzer: ETFAnalyzer instance (data should already be loaded)
            variance_threshold: Cumulative PC1 contribution threshold (default: 0.30)
        """
        self.analyzer = analyzer
        self.variance_threshold = variance_threshold
        self.start_date = '2018-01-01'  # Strict start date as requested
        self.pca_results = {}
        self.selected_stocks = []
        
    def run_pca_analysis(self, log_returns_df: pd.DataFrame) -> Dict:
        """
        Run PCA analysis on log returns matrix using Common Period strategy
        
        Args:
            log_returns_df: DataFrame with dates as index, stocks as columns, log returns as values
            
        Returns:
            dict: PCA results including components, explained variance, loadings
        """
        print("🔍 Running PCA analysis on log returns matrix...")
        print(f"   Input shape: {log_returns_df.shape} (dates × stocks)")
        
        # 1. Common Period Strategy: Slice from 2018-01-01
        start_date = '2018-01-01'
        log_returns_sliced = log_returns_df.loc[start_date:]
        
        if log_returns_sliced.empty:
            print(f"❌ No data found after {start_date}")
            return {}
            
        print(f"   Sliced to Common Period ({start_date}+): {log_returns_sliced.shape}")
        
        # 2. Drop columns (stocks) that have ANY missing data in this period
        # This prioritizes a clean matrix over a complete list of tickers
        log_returns_clean = log_returns_sliced.dropna(axis=1)
        
        print(f"   After dropping incomplete stocks: {log_returns_clean.shape} (dates × stocks)")
        print(f"   Removed {log_returns_sliced.shape[1] - log_returns_clean.shape[1]} stocks with missing data in common period")
        
        if log_returns_clean.empty:
            print("❌ No valid stocks remaining after cleaning")
            return {}
            
        # Standardize the log returns (each stock is a feature)
        scaler = StandardScaler()
        log_returns_scaled = scaler.fit_transform(log_returns_clean)
        
        # Run PCA - we only need the first component for the new logic, 
        # but we'll compute more to show the dominance of PC1
        n_components = min(20, log_returns_clean.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(log_returns_scaled)
        
        # Create loadings DataFrame (stocks × components)
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=log_returns_clean.columns  # Stock symbols
        )
        
        explained_variance = pca.explained_variance_ratio_
        
        self.pca_results = {
            'pca': pca,
            'scaler': scaler,
            'loadings': loadings_df,
            'explained_variance': explained_variance,
            'n_components': n_components,
            'log_returns_clean': log_returns_clean
        }
        
        print(f"✅ PCA completed. PC1 explains {explained_variance[0]*100:.1f}% of variance")
        
        return self.pca_results
    
    def select_stocks_by_variance(self, pca_results: Optional[Dict] = None) -> List[str]:
        """
        Select stocks based on PC1 Contribution Score
        
        Strategy:
        1. Isolate PC1 (Market Signal)
        2. Calculate Contribution Score = Loading^2 / Sum(Loadings^2)
        3. Sort by Score descending
        4. Select top stocks until cumulative score >= variance_threshold
        
        Args:
            pca_results: Optional PCA results dict (uses self.pca_results if None)
            
        Returns:
            list: Selected stock symbols
        """
        if pca_results is None:
            pca_results = self.pca_results
        
        if not pca_results:
            print("❌ No PCA results available. Run run_pca_analysis() first.")
            return []
        
        print(f"🎯 Selecting stocks to explain {self.variance_threshold*100:.1f}% of PC1 weights...")
        
        loadings_df = pca_results['loadings']
        
        # 1. Get PC1 Loadings
        pc1_loadings = loadings_df['PC1']
        
        # 2. Calculate Contribution Scores
        # Score = Loading^2 / Sum(Loadings^2)
        squared_loadings = pc1_loadings ** 2
        total_squared_loadings = squared_loadings.sum()
        contribution_scores = squared_loadings / total_squared_loadings
        
        # 3. Sort by Score descending
        sorted_stocks = contribution_scores.sort_values(ascending=False)
        
        # 4. Select stocks until cumulative score >= threshold
        cumulative_scores = sorted_stocks.cumsum()
        
        # Find cutoff index
        cutoff_mask = cumulative_scores >= self.variance_threshold
        if cutoff_mask.any():
            # Get the first index where threshold is met
            cutoff_idx = np.argmax(cutoff_mask.values)
            # Include that stock (slice is exclusive at end, so +1)
            selected_stocks = sorted_stocks.index[:cutoff_idx+1].tolist()
        else:
            # If threshold never met (unlikely), take all
            selected_stocks = sorted_stocks.index.tolist()
            
        self.selected_stocks = selected_stocks
        self.stock_scores = sorted_stocks  # Store for plotting
        
        print(f"✅ Selected {len(selected_stocks)} stocks to reach {cumulative_scores.iloc[len(selected_stocks)-1]*100:.1f}% of PC1")
        print("   Top 10 Contributors:")
        for stock in selected_stocks[:10]:
            print(f"   - {stock}: {sorted_stocks[stock]*100:.2f}% (Cum: {cumulative_scores[stock]*100:.1f}%)")
            
        return selected_stocks
    
    def evaluate_mimic_performance(self, 
                                   selected_stocks: List[str],
                                   etf_returns: pd.Series,
                                   stock_returns_df: pd.DataFrame) -> Dict:
        """
        Evaluate how well selected stocks mimic ETF behavior
        
        Args:
            selected_stocks: List of selected stock symbols
            etf_returns: Series of ETF returns (same date index as stock_returns_df)
            stock_returns_df: DataFrame with stock returns (dates × stocks)
            
        Returns:
            dict: Performance metrics including tracking error, R², correlation, beta, alpha
        """
        print(f"📊 Evaluating mimic performance for {len(selected_stocks)} stocks...")
        
        if not selected_stocks:
            return {
                'tracking_error': np.nan,
                'r2_score': 0.0,
                'mae': np.nan,
                'correlation': np.nan,
                'beta': np.nan,
                'alpha': np.nan,
                'coefficients': {}
            }
        
        # Align data
        aligned_data = pd.concat([stock_returns_df[selected_stocks], etf_returns], axis=1).dropna()
        X = aligned_data[selected_stocks]
        y = aligned_data[etf_returns.name]
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Predictions
        y_pred = reg.predict(X)
        
        # Calculate tracking error metrics
        residuals = y - y_pred
        tracking_error = np.std(residuals) * np.sqrt(252)  # Annualized
        r2 = r2_score(y, y_pred)
        mae = np.mean(np.abs(residuals))
        
        # Correlation
        correlation = np.corrcoef(y, y_pred)[0, 1]
        
        # Beta and Alpha
        covariance = np.cov(y, y_pred)[0, 1]
        etf_variance = np.var(y)
        beta = covariance / etf_variance if etf_variance > 0 else np.nan
        
        # Annualized returns for alpha calculation
        etf_annual_return = y.mean() * 252
        portfolio_annual_return = y_pred.mean() * 252
        alpha = portfolio_annual_return - beta * etf_annual_return if not np.isnan(beta) else np.nan
        
        results = {
            'tracking_error': tracking_error,
            'r2_score': r2,
            'mae': mae,
            'correlation': correlation,
            'beta': beta,
            'alpha': alpha,
            'coefficients': dict(zip(selected_stocks, reg.coef_)),
            'intercept': reg.intercept_
        }
        
        print(f"✅ Performance evaluation completed")
        print(f"   Tracking Error (annualized): {tracking_error:.3f}")
        print(f"   R² Score: {r2:.3f}")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   Beta: {beta:.3f}")
        print(f"   Alpha: {alpha:.4f}")
        
        return results
    
    def plot_pca_results(self, output_dir: str):
        """
        Plot PCA results: 4 separate images
        1. Pareto Chart (Individual Contributions)
        2. Cumulative Contribution
        3. Scree Plot (Explained Variance)
        4. Top Stocks Heatmap (Weights)
        
        Args:
            output_dir: Directory to save the plots
        """
        if not hasattr(self, 'stock_scores'):
            print("❌ No stock scores available. Run select_stocks_by_variance() first.")
            return
        
        print(f"📊 Creating PCA analysis plots in {output_dir}...")
        
        # Prepare data
        n_plot = 20  # Show top 20 for Pareto/Heatmap
        top_stocks = self.stock_scores.head(n_plot)
        cumulative_scores = self.stock_scores.cumsum()
        
        # --- Plot 1: Pareto Chart (Top 20) ---
        plt.figure(figsize=(12, 8))
        x = range(len(top_stocks))
        plt.bar(x, top_stocks.values, color='skyblue', alpha=0.7)
        plt.xticks(x, top_stocks.index, rotation=90, fontsize=8)
        plt.title(f'Top {n_plot} Stocks by PC1 Contribution')
        plt.ylabel('Contribution Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_1_pareto.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Plot 2: Cumulative Contribution (All Selected) ---
        plt.figure(figsize=(12, 8))
        # Plot up to the cutoff or a bit more
        n_cumulative = min(len(self.selected_stocks) + 20, len(self.stock_scores))
        cum_data = cumulative_scores.iloc[:n_cumulative]
        
        plt.plot(range(len(cum_data)), cum_data.values, 'r-', linewidth=2)
        plt.axhline(y=self.variance_threshold, color='green', linestyle='--', label=f'Threshold ({self.variance_threshold:.0%})')
        plt.axvline(x=len(self.selected_stocks)-1, color='black', linestyle=':', label='Cutoff')
        plt.title('Cumulative PC1 Contribution')
        plt.xlabel('Number of Stocks')
        plt.ylabel('Cumulative Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_2_cumulative.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Plot 3: Scree Plot (First 10 PCs) ---
        plt.figure(figsize=(12, 8))
        if self.pca_results:
            explained_var = self.pca_results['explained_variance'][:10]
            plt.bar(range(1, len(explained_var)+1), explained_var * 100, color='purple', alpha=0.7)
            plt.plot(range(1, len(explained_var)+1), explained_var * 100, 'ko-', linewidth=1)
            plt.title('Scree Plot: Explained Variance by Component')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance (%)')
            plt.xticks(range(1, len(explained_var)+1))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/pca_3_scree.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        # --- Plot 4: Heatmap of Weights (Top 20 Stocks in PC1) ---
        plt.figure(figsize=(10, 12))
        # Get loadings for top stocks
        loadings_df = self.pca_results['loadings']
        top_loadings = loadings_df.loc[top_stocks.index, ['PC1']]
        sns.heatmap(top_loadings, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
        plt.title(f'PC1 Loadings for Top {n_plot} Stocks')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_4_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 PCA analysis plots saved to {output_dir}/")
    
    def plot_stock_movements(self, selected_stocks: List[str], etf_returns: pd.Series, 
                            stock_returns_df: pd.DataFrame, etf_symbol: str, 
                            output_dir: str):
        """
        Plot selected stocks vs ETF and calculate representation quality
        Generates 4 separate images.
        
        Args:
            selected_stocks: List of selected stock symbols
            etf_returns: Series of ETF returns
            stock_returns_df: DataFrame with stock returns (dates × stocks)
            etf_symbol: ETF symbol for display
            output_dir: Directory to save the plots
        """
        print(f"📈 Creating stock movement analysis for {etf_symbol}...")
        
        # Calculate cumulative returns
        etf_cumulative = (1 + etf_returns).cumprod()
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(selected_stocks)))
        
        # --- Plot 1: Selected stocks vs ETF (Cumulative Returns) ---
        plt.figure(figsize=(14, 8))
        plt.plot(etf_cumulative.index, etf_cumulative.values, 'k-', linewidth=3, 
                       label=etf_symbol, alpha=0.9)
        
        for i, stock in enumerate(selected_stocks):
            if stock in stock_returns_df.columns:
                stock_cumulative = (1 + stock_returns_df[stock]).cumprod()
                plt.plot(stock_cumulative.index, stock_cumulative.values, 
                               color=colors[i % len(colors)], alpha=0.5, label=stock, linewidth=1)
        
        plt.title(f'Selected Stocks vs {etf_symbol} (Cumulative Returns)')
        plt.ylabel('Cumulative Return')
        # Legend outside, multi-column
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, ncol=3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/1_cumulative_returns.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Plot 2: Individual stock returns vs ETF (Normalized) ---
        plt.figure(figsize=(14, 8))
        plt.plot(etf_cumulative.index, etf_cumulative.values / etf_cumulative.iloc[0], 
                       'k-', linewidth=3, label=etf_symbol, alpha=0.9)
        
        for i, stock in enumerate(selected_stocks):
            if stock in stock_returns_df.columns:
                stock_cumulative = (1 + stock_returns_df[stock]).cumprod()
                normalized = stock_cumulative / stock_cumulative.iloc[0]
                plt.plot(normalized.index, normalized.values, 
                               color=colors[i % len(colors)], alpha=0.5, label=stock, linewidth=1)
        
        plt.title(f'Selected Stocks vs {etf_symbol} (Normalized)')
        plt.ylabel('Normalized Cumulative Return')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, ncol=3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_normalized_returns.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Plot 3: Correlation heatmap - selected stocks ---
        # Make figure large enough for all stocks
        heatmap_size = max(12, len(selected_stocks) * 0.3)
        plt.figure(figsize=(heatmap_size, heatmap_size))
        
        if selected_stocks:
            # Include ETF in correlation matrix
            data_for_corr = pd.concat([stock_returns_df[selected_stocks], etf_returns], axis=1)
            corr_data = data_for_corr.corr()
            
            sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.2f', cbar_kws={'label': 'Correlation'},
                       square=True, linewidths=0.5, annot_kws={"size": 8})
            plt.title(f'Correlation Matrix: Selected Stocks & {etf_symbol}')
        else:
            plt.text(0.5, 0.5, 'No stocks selected', ha='center', va='center')
            plt.title(f'Selected Stocks Correlation with {etf_symbol}')
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Plot 4: Rolling correlation with ETF ---
        plt.figure(figsize=(14, 8))
        window = 30  # 30-day rolling window
        rolling_corrs = {}
        
        for stock in selected_stocks:
            if stock in stock_returns_df.columns:
                rolling_corr = stock_returns_df[stock].rolling(window=window).corr(etf_returns)
                rolling_corrs[stock] = rolling_corr
        
        for i, (stock, rolling_corr) in enumerate(rolling_corrs.items()):
            plt.plot(rolling_corr.index, rolling_corr.values, 
                           color=colors[i % len(colors)], alpha=0.6, label=stock, linewidth=1)
        
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% Correlation')
        plt.title(f'Rolling Correlation with {etf_symbol} ({window}-day window)')
        plt.ylabel('Correlation')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, ncol=3)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/4_rolling_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Stock movement plots saved to {output_dir}/")
    
    def plot_threshold_sensitivity(self, save_path: Optional[str] = None):
        """
        Plot Number of Stocks vs Cumulative Explained Variance (10% to 90%)
        
        Args:
            save_path: Optional path to save the plot
        """
        if not hasattr(self, 'stock_scores'):
            print("❌ No stock scores available. Run select_stocks_by_variance() first.")
            return
            
        print("📈 Creating threshold sensitivity analysis...")
        
        cumulative_scores = self.stock_scores.cumsum()
        
        thresholds = np.arange(0.1, 1.0, 0.1)  # 0.1, 0.2, ..., 0.9
        stock_counts = []
        
        for t in thresholds:
            # Find number of stocks needed to reach threshold t
            cutoff_mask = cumulative_scores >= t
            if cutoff_mask.any():
                count = np.argmax(cutoff_mask.values) + 1
            else:
                count = len(cumulative_scores)
            
            stock_counts.append(count)
            
        # Create Plot
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds * 100, stock_counts, 'bo-', linewidth=2, markersize=8)
        
        # Add labels for each point
        for x, y in zip(thresholds * 100, stock_counts):
            plt.annotate(f'{y}', 
                        (x, y), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        fontweight='bold')
            
        plt.title('Stock Count vs Explained Variance Target')
        plt.xlabel('Target Cumulative Explained Variance (%)')
        plt.ylabel('Number of Stocks Required')
        plt.grid(True, alpha=0.3)
        plt.xticks(thresholds * 100)
        
        # Highlight current selection
        current_pct = self.variance_threshold * 100
        current_count = len(self.selected_stocks)
        plt.plot(current_pct, current_count, 'r*', markersize=15, label=f'Current ({current_pct:.0f}%, {current_count})')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Threshold sensitivity plot saved to {save_path}")
            
        plt.close()
    
    def create_synthetic_etf(self, selected_stocks: List[str], stock_returns_df: pd.DataFrame) -> pd.Series:
        """
        Create a synthetic ETF from selected stocks (equal-weighted portfolio)
        
        Args:
            selected_stocks: List of selected stock symbols
            stock_returns_df: DataFrame with stock returns (dates × stocks)
            
        Returns:
            Series: Synthetic ETF returns
        """
        if not selected_stocks:
            return pd.Series(index=stock_returns_df.index, name="Synthetic_ETF")
        
        # Create equal-weight portfolio
        portfolio_returns = stock_returns_df[selected_stocks].mean(axis=1)
        portfolio_returns.name = "Synthetic_ETF"
        
        return portfolio_returns
    
    def plot_synthetic_etf(self, selected_stocks: List[str], etf_returns: pd.Series,
                          stock_returns_df: pd.DataFrame, etf_symbol: str,
                          output_dir: str):
        """
        Plot synthetic ETF created from selected stocks vs actual ETF
        Generates 4 separate images.
        
        Args:
            selected_stocks: List of selected stock symbols
            etf_returns: Series of ETF returns
            stock_returns_df: DataFrame with stock returns (dates × stocks)
            etf_symbol: ETF symbol for display
            output_dir: Directory to save the plots
        """
        print(f"📈 Creating synthetic ETF analysis for {etf_symbol}...")
        
        # Create synthetic ETF
        synthetic_etf = self.create_synthetic_etf(selected_stocks, stock_returns_df)
        
        # Calculate cumulative returns
        etf_cumulative = (1 + etf_returns).cumprod()
        synthetic_cumulative = (1 + synthetic_etf).cumprod()
        
        # --- Plot 1: Cumulative returns comparison ---
        plt.figure(figsize=(14, 8))
        plt.plot(etf_cumulative.index, etf_cumulative.values, 'k-', linewidth=3, 
                       label=etf_symbol, alpha=0.8)
        plt.plot(synthetic_cumulative.index, synthetic_cumulative.values, 'b-', 
                       linewidth=2, label='Synthetic ETF (PCA Selected)', alpha=0.8)
        
        plt.title(f'Synthetic ETF vs {etf_symbol} (Cumulative Returns)')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/synth_1_cumulative.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Plot 2: Rolling correlation with ETF ---
        plt.figure(figsize=(14, 8))
        window = 30  # 30-day rolling window
        
        synthetic_corr = synthetic_etf.rolling(window=window).corr(etf_returns)
        
        plt.plot(synthetic_corr.index, synthetic_corr.values, 'b-', linewidth=2, 
                       label='Synthetic ETF Correlation', alpha=0.8)
        plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% Correlation')
        plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% Correlation')
        
        plt.title(f'Rolling Correlation with {etf_symbol} ({window}-day window)')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/synth_2_rolling_corr.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Plot 3: Tracking error over time ---
        plt.figure(figsize=(14, 8))
        tracking_error = (synthetic_etf - etf_returns).rolling(window=window).std() * np.sqrt(252)
        
        plt.plot(tracking_error.index, tracking_error.values, 'b-', linewidth=2, 
                       label='Synthetic ETF Tracking Error', alpha=0.8)
        
        plt.title(f'Rolling Tracking Error ({window}-day window)')
        plt.ylabel('Annualized Tracking Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/synth_3_tracking_error.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # --- Plot 4: Performance metrics comparison ---
        plt.figure(figsize=(12, 8))
        metrics = ['Total Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Beta']
        
        # Calculate metrics
        etf_total_return = (etf_cumulative.iloc[-1] - 1) * 100
        synthetic_total_return = (synthetic_cumulative.iloc[-1] - 1) * 100
        
        etf_vol = etf_returns.std() * np.sqrt(252) * 100
        synthetic_vol = synthetic_etf.std() * np.sqrt(252) * 100
        
        etf_sharpe = etf_returns.mean() * 252 / (etf_returns.std() * np.sqrt(252)) if etf_returns.std() > 0 else 0
        synthetic_sharpe = synthetic_etf.mean() * 252 / (synthetic_etf.std() * np.sqrt(252)) if synthetic_etf.std() > 0 else 0
        
        # Max drawdown
        etf_dd = ((etf_cumulative / etf_cumulative.expanding().max()) - 1).min() * 100
        synthetic_dd = ((synthetic_cumulative / synthetic_cumulative.expanding().max()) - 1).min() * 100
        
        # Beta
        aligned_data = pd.concat([synthetic_etf, etf_returns], axis=1).dropna()
        if len(aligned_data) > 0:
            synthetic_beta = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])[0, 1] / np.var(aligned_data.iloc[:, 1])
        else:
            synthetic_beta = np.nan
        
        etf_values = [etf_total_return, etf_vol, etf_sharpe, etf_dd, 1.0]
        synthetic_values = [synthetic_total_return, synthetic_vol, synthetic_sharpe, synthetic_dd, synthetic_beta]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, etf_values, width, label=etf_symbol, alpha=0.8)
        plt.bar(x + width/2, synthetic_values, width, label='Synthetic ETF', alpha=0.8)
        
        plt.title('Performance Metrics Comparison')
        plt.ylabel('Value')
        plt.xticks(x, metrics, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/synth_4_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Synthetic ETF analysis plots saved to {output_dir}/")
        
        # Print summary statistics
        print(f"\n📊 SYNTHETIC ETF PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{etf_symbol}:")
        print(f"   • Total Return: {etf_total_return:.1f}%")
        print(f"   • Volatility: {etf_vol:.1f}%")
        print(f"   • Sharpe Ratio: {etf_sharpe:.2f}")
        print(f"   • Max Drawdown: {etf_dd:.1f}%")
        
        print(f"\nSynthetic ETF ({len(selected_stocks)} stocks):")
        print(f"   • Total Return: {synthetic_total_return:.1f}%")
        print(f"   • Volatility: {synthetic_vol:.1f}%")
        print(f"   • Sharpe Ratio: {synthetic_sharpe:.2f}")
        print(f"   • Max Drawdown: {synthetic_dd:.1f}%")
        print(f"   • Beta: {synthetic_beta:.2f}")
        if len(aligned_data) > 0:
            print(f"   • Avg Correlation with {etf_symbol}: {aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1]):.3f}")
        
        return {
            'synthetic_etf': synthetic_etf,
            'synthetic_cumulative': synthetic_cumulative,
            'correlation': aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1]) if len(aligned_data) > 0 else np.nan
        }
    
    def run_complete_analysis(self, etf_symbol: str) -> Dict:
        """
        Run complete ETF mimicking analysis pipeline
        
        Args:
            etf_symbol: ETF symbol to analyze
            
        Returns:
            dict: Complete analysis results
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Complete ETF Mimicking Analysis for {etf_symbol}")
        print(f"{'='*60}")
        
        # Step 1: Get log returns matrix from analyzer
        print(f"\n📊 Step 1: Getting log returns matrix...")
        # Step 1: Get data and create log returns matrix
        print(f"\n📊 Step 1: Getting stock data and creating log returns matrix...")
        
        # Use new flatten_for_pca method
        flat_df = self.analyzer.flatten_for_pca(etf_symbol)
        
        if flat_df.empty:
            print(f"❌ No data available for {etf_symbol}")
            return {}
            
        # Pivot to get Close prices (Dates x Stocks)
        # flat_df has columns: Date, Stock_Symbol, Close, ...
        close_prices = flat_df.pivot(index='Date', columns='Stock_Symbol', values='Close')
        
        # Calculate log returns: log(1 + pct_change)
        log_returns_df = np.log(1 + close_prices.pct_change())
        
        # Drop the first row (NaN) and any columns that are all NaN
        log_returns_df = log_returns_df.dropna(how='all').dropna(axis=1, how='all')
        
        if log_returns_df.empty:
            print(f"❌ No log returns data available for {etf_symbol}")
            return {}
        
        # Step 2: Run PCA analysis
        print(f"\n📊 Step 2: Running PCA analysis...")
        pca_results = self.run_pca_analysis(log_returns_df)
        
        # Step 3: Select stocks by variance
        print(f"\n📊 Step 3: Selecting stocks...")
        selected_stocks = self.select_stocks_by_variance(pca_results)
        
        # Step 4: Fetch ETF returns for comparison
        print(f"\n📊 Step 4: Fetching ETF returns...")
        etf_returns = fetch_etf_returns(etf_symbol, start=self.start_date)
        
        # Convert log returns back to regular returns for comparison
        stock_returns_df = np.exp(log_returns_df) - 1
        
        # Align dates
        common_dates = log_returns_df.index.intersection(etf_returns.index)
        stock_returns_df = stock_returns_df.loc[common_dates]
        etf_returns = etf_returns.loc[common_dates]
        
        # Step 5: Evaluate mimic performance
        print(f"\n📊 Step 5: Evaluating mimic performance...")
        performance = self.evaluate_mimic_performance(selected_stocks, etf_returns, stock_returns_df)
        
        # Step 6: Create plots
        print(f"\n📊 Step 6: Creating visualization plots...")
        output_dir = create_output_dir(etf_symbol)
        
        # Plot PCA results (saves 4 separate images)
        self.plot_pca_results(output_dir)
        
        # Plot stock movements (saves 4 separate images)
        self.plot_stock_movements(selected_stocks, etf_returns, stock_returns_df, 
                                 etf_symbol, output_dir)
        
        # Plot synthetic ETF (saves 4 separate images)
        synthetic_results = self.plot_synthetic_etf(selected_stocks, etf_returns, 
                                                   stock_returns_df, etf_symbol,
                                                   output_dir)
        
        # Plot threshold sensitivity
        self.plot_threshold_sensitivity(f'{output_dir}/threshold_sensitivity.png')
        
        # Compile results
        results = {
            'etf_symbol': etf_symbol,
            'log_returns_matrix': log_returns_df,
            'pca_results': pca_results,
            'selected_stocks': selected_stocks,
            'performance': performance,
            'etf_returns': etf_returns,
            'stock_returns_df': stock_returns_df,
            'synthetic_etf': synthetic_results
        }
        
        print(f"\n✅ Complete analysis finished for {etf_symbol}!")
        print(f"   Selected {len(selected_stocks)} stocks: {selected_stocks}")
        print(f"   Tracking Error: {performance['tracking_error']:.3f}")
        print(f"   R² Score: {performance['r2_score']:.3f}")
        print(f"   Plots saved to: {output_dir}/")
        
        return results


def main():
    """Main function to run ETF mimicking analysis"""
    print("🚀 Starting ETF Mimicking Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ETFAnalyzer(
        holdings_dir='src/holdings_data',  # Relative to project root
        data_period='10y',  # Need enough history for 2018 start date
        max_stocks_per_etf=None,  # None = use ALL stocks from holdings CSV
        headless=True
    )
    
    # Run complete analysis to get stock data
    etf_symbols = ['SPY', 'VTI', 'VOO', 'QQQ']
    #etf_symbols = ['SPY']
    analyzer.run_complete_analysis(etf_symbols, force_download=False)
    
    # Run mimicking analysis for each ETF
    mimic_analyzer = MimicETF(analyzer)
    all_results = {}
    
    for etf_symbol in etf_symbols:
        print(f"\n🚀 Running analysis for {etf_symbol} with threshold={mimic_analyzer.variance_threshold}...")
        try:
            results = mimic_analyzer.run_complete_analysis(etf_symbol)
            all_results[etf_symbol] = results
        except Exception as e:
            print(f"❌ Error analyzing {etf_symbol}: {e}")
    
    return all_results


if __name__ == "__main__":
    results = main()
    print(results)