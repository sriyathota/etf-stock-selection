"""
Financial indicator features (Group 2 - Feature Engineering)

Builds a daily (date x ticker) panel of fundamental features from yfinance:
    pe_ratio, pb_ratio, eps_ttm, eps_growth_yoy, roe, profit_margin, debt_to_equity

Point-in-time rule: a report only becomes visible QUARTERLY_LAG / ANNUAL_LAG days after its
period end (companies file 10-Qs ~40 days and 10-Ks ~60-90 days after period end).
This keeps the features usable for out-of-sample / walk-forward evaluation.
Two leaks remain that yfinance cannot avoid: figures are the latest restated values, and
Q4 numbers come from the 10-K, which some smaller firms file more than 45 days after year end.

TTM EPS sums four quarterly diluted EPS figures (the usual approximation; it differs slightly
from annual EPS when the share count changes). ROE uses period-end equity, not average equity.
yfinance restates EPS and share counts for splits, matching the split-adjusted Close, so
P/E and P/B don't jump on split dates (checked on NVDA's 10:1 split, 2024-06-10).

Limitation: yfinance only returns ~4 annual and ~5 quarterly statements, so
features are NaN before roughly 4 years ago. SEC EDGAR "companyfacts" is the
free source for longer history.

Usage (from repo root):
    python src/features/financial_indicators.py --tickers AAPL MSFT NVDA
    python src/features/financial_indicators.py --etf QQQ --out data/features/financial_QQQ.csv
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from align import normalize_ticker           # run as a script from src/features
except ImportError:
    from src.features.align import normalize_ticker

# Days between a period's end and the date its numbers are assumed public (see point-in-time rule)
QUARTERLY_LAG = pd.Timedelta(days=45)
ANNUAL_LAG = pd.Timedelta(days=90)

FEATURES = ["pe_ratio", "pb_ratio", "eps_ttm", "eps_growth_yoy",
            "roe", "profit_margin", "debt_to_equity"]


def _row(df: pd.DataFrame, *names) -> pd.Series:
    """First statement line that exists, oldest period first.

    yfinance labels the same line differently across companies, so callers pass fallbacks in order.
    Returns an empty Series when none exist, which becomes an all-NaN column downstream.
    """
    for name in names:
        if name in df.index:
            return df.loc[name].astype(float).sort_index()
    return pd.Series(dtype=float)


def _annual_reports(t: yf.Ticker) -> pd.DataFrame:
    """One row per fiscal year (index = period end), plus the date it becomes visible."""
    inc, bal = t.income_stmt, t.balance_sheet
    if inc.empty or bal.empty:
        return pd.DataFrame()
    rep = pd.DataFrame({
        "eps": _row(inc, "Diluted EPS", "Basic EPS"),
        "net_income": _row(inc, "Net Income", "Net Income Common Stockholders"),
        "revenue": _row(inc, "Total Revenue", "Operating Revenue"),
        "equity": _row(bal, "Stockholders Equity", "Common Stock Equity"),
        "debt": _row(bal, "Total Debt"),
        "shares": _row(bal, "Ordinary Shares Number", "Share Issued"),
    })
    rep["available"] = rep.index + ANNUAL_LAG
    return rep


def _quarterly_reports(t: yf.Ticker) -> pd.DataFrame:
    """Trailing-twelve-month figures from the last 4 quarters of each report."""
    inc, bal = t.quarterly_income_stmt, t.quarterly_balance_sheet
    if inc.empty or bal.empty:
        return pd.DataFrame()
    eps =_row(inc, "Diluted EPS", "Basic EPS")
    # Only trust a 4-quarter sum when the quarters are consecutive (~9 months from first to last end)
    span = pd.Series(eps.index, index=eps.index).diff(3)
    consecutive = span.between(pd.Timedelta(days=250), pd.Timedelta(days=300))
    ttm = lambda s: s.rolling(4).sum().where(consecutive)
    # Income-statement lines are flows, so they are summed to TTM; balance-sheet lines are
    # point-in-time stocks, so the latest quarter's value is used as-is
    rep = pd.DataFrame({
        "eps": ttm(eps),
        "net_income": ttm(_row(inc, "Net Income", "Net Income Common Stockholders")),
        "revenue": ttm(_row(inc, "Total Revenue", "Operating Revenue")),
        "equity": _row(bal, "Stockholders Equity", "Common Stock Equity"),
        "debt": _row(bal, "Total Debt"),
        "shares": _row(bal, "Ordinary Shares Number", "Share Issued"),
    }).dropna(subset=["eps"])
    rep["available"] = rep.index + QUARTERLY_LAG
    return rep


def fundamentals_for_ticker(symbol: str, start: str = "2018-01-01") -> pd.DataFrame:
    """Daily financial-indicator features for one ticker (index = date)."""
    t = yf.Ticker(normalize_ticker(symbol))
    # Annual and quarterly reports share one timeline; each row is "what we knew from `available` on"
    reports = pd.concat([_annual_reports(t), _quarterly_reports(t)])
    if reports.empty:
        return pd.DataFrame()

    # Latest report wins when an annual and a quarterly report become visible together
    reports = (reports.rename_axis("period_end").reset_index()
               .sort_values(["available", "period_end"])
               .drop_duplicates("available", keep="last"))

    # YoY EPS growth: compare against the report whose period ended ~1 year earlier
    reports["eps_prev_year"] = [
        _eps_one_year_before(reports, row.period_end) for row in reports.itertuples()
    ]

    prices = t.history(start=start, auto_adjust=False)["Close"]
    if prices.empty:
        return pd.DataFrame()
    # Strip the exchange timezone so the index matches the naive `available` dates
    prices.index = prices.index.tz_localize(None).normalize().astype("datetime64[ns]")
    reports["available"] = reports["available"].astype("datetime64[ns]")

    # Each trading day takes the most recent report already public on that day
    daily = pd.merge_asof(prices.rename("close").to_frame(), reports.set_index("available"),
                          left_index=True, right_index=True, direction="backward")

    # Ratios are left NaN when the denominator makes them meaningless (negative equity, losses, ...)
    out = pd.DataFrame(index=daily.index)
    bvps = daily.equity / daily.shares
    out["pe_ratio"] = (daily.close / daily.eps).where(daily.eps > 0)        # undefined for losses
    out["pb_ratio"] = (daily.close / bvps).where(bvps > 0)
    out["eps_ttm"] = daily.eps
    prev = daily.eps_prev_year
    out["eps_growth_yoy"] = ((daily.eps - prev) / prev.abs()).where(prev.abs() >= 0.05)  # unstable near 0
    out["roe"] = (daily.net_income / daily.equity).where(daily.equity > 0)
    out["profit_margin"] = (daily.net_income / daily.revenue).where(daily.revenue > 0)
    out["debt_to_equity"] = (daily.debt / daily.equity).where(daily.equity > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def _eps_one_year_before(reports: pd.DataFrame, period_end: pd.Timestamp) -> float:
    """EPS of the report whose period ended within 20 days of one year before `period_end`, else NaN."""
    target = period_end - pd.DateOffset(years=1)
    window = reports[(reports.period_end - target).abs() <= pd.Timedelta(days=20)]
    return window.eps.iloc[-1] if len(window) else np.nan


def build_financial_features(tickers, start: str = "2018-01-01", delay: float = 0.2) -> pd.DataFrame:
    """Long panel with columns: date, ticker, <FEATURES>."""
    frames = []
    for i, sym in enumerate(tickers, 1):
        try:
            df = fundamentals_for_ticker(sym, start)
        except Exception as e:                  # one bad ticker shouldn't stop a 100-stock run
            print(f"❌ {sym}: {e}")
            continue
        if df.empty:
            print(f"❌ {sym}: no fundamentals")
            continue
        frames.append(df.assign(ticker=normalize_ticker(sym)).rename_axis("date").reset_index())
        print(f"✅ [{i}/{len(tickers)}] {sym}: {df.dropna(how='all').shape[0]} days with data")
        time.sleep(delay)                       # stay under Yahoo's rate limit
    if not frames:
        return pd.DataFrame(columns=["date", "ticker"] + FEATURES)
    return pd.concat(frames, ignore_index=True)[["date", "ticker"] + FEATURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--etf", help="use all holdings from src/holdings_data/<ETF>-holdings.csv")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out", default="data/features/financial_indicators.csv")
    args = ap.parse_args()

    tickers = args.tickers
    if args.etf:
        tickers = pd.read_csv(f"src/holdings_data/{args.etf}-holdings.csv")["Symbol"].dropna().tolist()
        tickers = [s for s in tickers if s != "--"]      # holdings files use "--" for cash/other rows
    if not tickers:
        ap.error("pass --tickers or --etf")

    panel = build_financial_features(tickers, args.start)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    panel.to_csv(args.out, index=False)
    print(f"\nsaved {panel.shape} -> {args.out}")
    print(panel.dropna(subset=["pe_ratio"]).groupby("ticker").tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
