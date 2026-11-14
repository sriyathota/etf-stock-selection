"""
Feature alignment: merges every feature group into one (date, ticker) dataset for Group 1.

Expected inputs (CSV, one per feature group):
    stock-level  -> columns: date, ticker, <features...>   (price-based, financial indicators)
    market-level -> columns: date, <features...>           (VIX, rates, S&P 500 returns, sector ETFs)

Market-level features have no ticker column. Each stock row takes the latest market value
dated on or before its own date (within market_tolerance_days), so different calendars
(holidays, FRED series) still line up. Market inputs must be stamped with the date the value
became KNOWN; a series published the next morning must be shifted by the producer, or it leaks.

Stock-level features are NOT forward-filled. Producers must already deliver daily rows, and a
NaN means "undefined" (e.g. P/E for a loss-making firm), so filling it would invent values.

Tickers are normalized to Yahoo form (BRK/B and BRK.B both become BRK-B) so files agree.

Usage (from repo root):
    python src/features/align.py \
        --stock data/features/financial_indicators.csv data/features/price_features.csv \
        --market data/features/market_indicators.csv \
        --out data/features/all_features.csv
"""

import argparse
import os
import warnings
from typing import Sequence

import pandas as pd


def _load(path: str) -> pd.DataFrame:
    """Read a feature CSV and turn its date column into plain, timezone-free midnight dates."""
    df = pd.read_csv(path)
    # Files may mix tz-aware and naive timestamps; going through UTC makes them comparable
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df["date"] = df["date"].astype("datetime64[ns]")
    return df


def normalize_ticker(sym: str) -> str:
    """Yahoo-style symbol: upper case, share-class separator written as '-' (BRK.B -> BRK-B)."""
    return str(sym).strip().upper().replace("/", "-").replace(".", "-")


def align_features(stock_frames: Sequence[pd.DataFrame],
                   market_frames: Sequence[pd.DataFrame] = (),
                   market_tolerance_days: int = 7) -> pd.DataFrame:
    """Outer-join stock-level frames on (date, ticker), then attach market-level frames as of each date."""
    if not stock_frames:
        raise ValueError("need at least one stock-level feature frame")

    # Stock-level groups: outer join so a row survives even if only one group has data for it
    panel, ticker_sets = None, []
    for df in stock_frames:
        df = df.assign(ticker=df["ticker"].map(normalize_ticker))
        if df.duplicated(["date", "ticker"]).any():
            raise ValueError("stock feature frame has duplicate (date, ticker) rows")
        ticker_sets.append(set(df["ticker"]))
        if panel is not None:
            clash = (set(df.columns) - {"date", "ticker"}) & set(panel.columns)
            if clash:
                raise ValueError(f"duplicate feature columns across groups: {clash}")
        panel = df if panel is None else panel.merge(df, on=["date", "ticker"], how="outer")

    # Tickers present in some files but not others usually mean a naming mismatch or a failed download
    if len(ticker_sets) > 1:
        unmatched = set.union(*ticker_sets) - set.intersection(*ticker_sets)
        if unmatched:
            warnings.warn(f"{len(unmatched)} tickers missing from some feature files, e.g. {sorted(unmatched)[:5]}")

    # Market-level groups: merge_asof needs both sides sorted by the join key
    panel = panel.sort_values("date")
    for df in market_frames:
        if df["date"].duplicated().any():
            raise ValueError("market feature frame has duplicate dates")
        clash = (set(df.columns) - {"date"}) & set(panel.columns)
        if clash:
            raise ValueError(f"market feature names clash with stock features: {clash}")
        # "backward" = only values dated on/before the stock row (no lookahead); stale after the tolerance
        panel = pd.merge_asof(panel, df.sort_values("date"), on="date", direction="backward",
                              tolerance=pd.Timedelta(days=market_tolerance_days))

    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", nargs="+", required=True, help="CSV(s) with date, ticker, features")
    ap.add_argument("--market", nargs="*", default=[], help="CSV(s) with date, features")
    ap.add_argument("--out", default="data/features/all_features.csv")
    args = ap.parse_args()

    panel = align_features([_load(p) for p in args.stock], [_load(p) for p in args.market])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    panel.to_csv(args.out, index=False)
    print(f"saved {panel.shape} -> {args.out}")
    print("non-null share per feature:")
    print(panel.drop(columns=["date", "ticker"]).notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
