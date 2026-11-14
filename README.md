# ETF Mimicking via PCA

This project implements a quantitative finance pipeline to create "synthetic" versions of major ETFs (like SPY, VTI, QQQ) using a subset of their constituent stocks. It leverages **Principal Component Analysis (PCA)** to identify the primary market factors and selects stocks that have the highest contribution to the first principal component (PC1).

## Project Overview

The core idea is that the first principal component of a broad market index typically represents the "Market Factor" (beta). By selecting stocks with the highest contribution to this factor, we can construct a smaller portfolio that highly correlates with the original ETF.

### Key Features
*   **Data Scraping**: Automatically downloads current ETF holdings from Schwab.
*   **Historical Data**: Fetches historical price data using `yfinance`.
*   **PCA Analysis**: Performs PCA on log returns to extract market signals.
*   **Smart Selection**: Selects stocks based on their cumulative contribution to PC1 (explained variance).
*   **Visualization**: Generates detailed plots for PCA results, correlation heatmaps, and synthetic ETF performance.

## Repository Structure

The core logic resides in the `src/` directory:

*   **`src/mimic_etf.py`**: The main entry point. Orchestrates the entire analysis, from data loading to PCA and visualization.
*   **`src/etf_analyzer.py`**: A helper class that manages data fetching, processing, and storage. Acts as the bridge between raw data and the analysis logic.
*   **`src/holdings_downloader.py`**: Handles scraping of ETF holdings data from Schwab using Selenium and Chrome.
*   **`src/holdings_csv_reader.py`**: Utilities for reading and parsing the downloaded holdings CSV files.
*   **`src/yfinance_data_fetcher.py`**: Wrapper around `yfinance` to fetch historical stock data efficiently.

## Installation & Dependencies

It is highly recommended to use a **virtual environment** to manage dependencies.

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd stock-selection
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate it (Mac/Linux)
    source venv/bin/activate

    # Activate it (Windows)
    # venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **System Requirements**:
    *   Google Chrome (for the Selenium scraper).

## How to Run

The project is designed to be run from the root directory.

### Run the Full Analysis
To run the analysis for the default ETFs (SPY, VTI, VOO, QQQ):

```bash
python src/mimic_etf.py
```

This will:
1.  Download the latest holdings for each ETF (if not already cached).
2.  Fetch 10 years of historical data for all constituents.
3.  Perform PCA to identify the top contributing stocks.
4.  Select a subset of stocks (default: top 25% cumulative PC1 weight).
5.  Generate a "Synthetic ETF" and compare its performance to the actual ETF.
6.  Save all results and plots to `src/pca_results/<ETF_SYMBOL>/`.

## Outputs

For each analyzed ETF, a folder is created in `src/pca_results/<ETF_SYMBOL>/` containing:

### 1. PCA Analysis
*   **`pca_1_pareto.png`**: Top 20 stocks contributing to the market factor.
*   **`pca_2_cumulative.png`**: Cumulative contribution of selected stocks.
*   **`pca_3_scree.png`**: Explained variance by each principal component.
*   **`pca_4_heatmap.png`**: Heatmap of PC1 loadings for top stocks.

### 2. Stock Movements
*   **`1_cumulative_returns.png`**: Cumulative returns of all selected stocks.
*   **`2_normalized_returns.png`**: Returns normalized to start at 1.0.
*   **`3_correlation_heatmap.png`**: Correlation matrix of the selected stocks.
*   **`4_rolling_correlation.png`**: Average rolling correlation among stocks.

### 3. Synthetic ETF Performance
*   **`synth_1_cumulative.png`**: Comparison of the Synthetic ETF vs. the Actual ETF.
*   **`synth_2_rolling_corr.png`**: Rolling correlation between Synthetic and Actual.
*   **`synth_3_tracking_error.png`**: Rolling tracking error (standard deviation of return differences).
*   **`synth_4_metrics.png`**: Key performance metrics (Alpha, Beta, R-Squared, etc.).
*   **`threshold_sensitivity.png`**: Analysis of how many stocks are needed to explain X% of variance.

## Configuration

You can adjust the analysis parameters in `src/mimic_etf.py`:

*   **`variance_threshold`**: Controls how many stocks are selected. Default is `0.25` (25% of cumulative PC1 weight), which typically selects ~70 stocks for SPY.
*   **`data_period`**: Length of historical data to fetch (default: '10y').
*   **`start_date`**: The analysis strictly filters for data from `2018-01-01` onwards to ensure a consistent time window.
