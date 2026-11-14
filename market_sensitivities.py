"""
Market sensitivity features (Group 2 - Feature Engineering)

Turns the date-level market indicators into stock-level features: each stock's rolling exposure
to market-wide drivers, so cross-sectional models (t-SNE, clustering, PCA on features) can tell
rate-sensitive utilities from high-beta tech. Output is a (date, ticker) panel.

Trailing-window OLS regressions per stock on daily log returns:
    market:  r = a + beta_mkt*spx_ret_1d                                  (classic beta)
    drivers: r = a + b*spx_ret_1d + beta_<x>*<driver>, one regression per driver:
             beta_vix (vix_chg_1d, per VIX point), beta_rate (rate_10y_chg_1d, per percentage point),
             beta_credit (credit_ret_1d), beta_dollar (dollar_ret_1d), beta_oil (oil_ret_1d)
    factor:  r - rf = a + load_mkt*ff_mkt_rf + load_smb*ff_smb + load_hml*ff_hml
                   + load_rmw*ff_rmw + load_cma*ff_cma + load_mom*ff_mom
             plus idio_vol (annualized residual std) and factor_r2.
Each driver beta is the exposure beyond what the market move explains. The drivers are not put in one
joint regression: VIX changes and HYG-IEF returns track the S&P so closely that a joint fit trades
them off against beta_mkt (it gave XOM and BRK-B negative market betas).

Point-in-time: the window ends on the row's own date, so each value uses only closes up to that day.
Only same-day (lag-0) market series are regressors: a regression pairs a stock's return with the
market move of the same session, and lagged series (fed funds, BAA, NFCI) are stamped by release date.
Fama-French data ends ~1 month before today, so factor loadings are NaN for that last stretch.

Usage (from repo root):
    python src/features/market_sensitivities.py --tickers AAPL MSFT XOM
    python src/features/market_sensitivities.py --etf SPY --out data/features/market_sensitivities_SPY.csv
    # reuse an existing market file (it must start >= `window` trading days before --start):
    python src/features/market_sensitivities.py --etf QQQ --market data/features/market_indicators.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from market_indicators import WARMUP, build_market_indicators   # run as a script from src/features
except ImportError:
    from src.features.market_indicators import WARMUP, build_market_indicators

MARKET = "spx_ret_1d"
DRIVERS = {"beta_vix": "vix_chg_1d", "beta_rate": "rate_10y_chg_1d", "beta_credit": "credit_ret_1d",
           "beta_dollar": "dollar_ret_1d", "beta_oil": "oil_ret_1d"}
FACTOR = {"load_mkt": "ff_mkt_rf", "load_smb": "ff_smb", "load_hml": "ff_hml",
          "load_rmw": "ff_rmw", "load_cma": "ff_cma", "load_mom": "ff_mom"}
FEATURES = ["beta_mkt"] + list(DRIVERS) + list(FACTOR) + ["idio_vol", "factor_r2"]


def normalize_ticker(sym: str) -> str:
    """Yahoo form: BRK/B and BRK.B both become BRK-B."""
    return str(sym).strip().upper().replace("/", "-").replace(".", "-")


def rolling_ols(y: np.ndarray, X: np.ndarray, window: int, min_obs: int):
    """Trailing-window OLS of y on [1, X] at every row, skipping rows with any NaN.

    Returns (coefs (T, k) without the intercept, residual std (T,), r2 (T,)). A row is NaN unless
    its own y and X are present and its window holds at least `min_obs` complete rows.
    """
    T, k = X.shape
    Z = np.column_stack([np.ones(T), X])
    ok = np.isfinite(y) & np.isfinite(Z).all(axis=1)
    Z, y = np.where(ok[:, None], Z, 0.0), np.where(ok, y, 0.0)

    def window_sum(a):
        c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
        out = np.full(a.shape, np.nan)
        out[window - 1:] = c[window:] - c[:-window]
        return out

    n = window_sum(ok.astype(float))
    ZZ = window_sum(Z[:, :, None] * Z[:, None, :])
    Zy = window_sum(Z * y[:, None])
    yy, ysum = window_sum(y ** 2), window_sum(y)

    coefs, resid_std, r2 = np.full((T, k), np.nan), np.full(T, np.nan), np.full(T, np.nan)
    valid = ok & (np.nan_to_num(n) >= max(min_obs, k + 2))
    if not valid.any():
        return coefs, resid_std, r2
    b = (np.linalg.pinv(ZZ[valid]) @ Zy[valid][:, :, None])[:, :, 0]
    ssr = np.clip(yy[valid] - (b * Zy[valid]).sum(axis=1), 0, None)
    sst = yy[valid] - ysum[valid] ** 2 / n[valid]
    coefs[valid] = b[:, 1:]
    resid_std[valid] = np.sqrt(ssr / (n[valid] - k - 1))
    r2[valid] = np.where(sst > 0, 1 - ssr / sst, np.nan)
    return coefs, resid_std, r2


def sensitivities_for_returns(r: pd.Series, market: pd.DataFrame, window: int, min_obs: int) -> pd.DataFrame:
    """Rolling macro betas and factor loadings for one stock's daily log returns (index = date)."""
    y = r.reindex(market.index).to_numpy(float)
    out = pd.DataFrame(index=market.index)

    mkt = market[MARKET].to_numpy(float)
    out["beta_mkt"] = rolling_ols(y, mkt[:, None], window, min_obs)[0][:, 0]
    for name, driver in DRIVERS.items():
        X = np.column_stack([mkt, market[driver].to_numpy(float)])
        out[name] = rolling_ols(y, X, window, min_obs)[0][:, 1]

    excess = y - market["ff_rf"].to_numpy(float)
    coefs, resid_std, r2 = rolling_ols(excess, market[list(FACTOR.values())].to_numpy(float), window, min_obs)
    out[list(FACTOR)] = coefs
    out["idio_vol"] = resid_std * np.sqrt(252)
    out["factor_r2"] = r2
    return out


def load_market(path: str = None, start: str = "2018-01-01") -> pd.DataFrame:
    if path:
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        df.index = df.index.as_unit("ns")
        return df
    market, failures = build_market_indicators((pd.Timestamp(start) - WARMUP).strftime("%Y-%m-%d"))
    needed = {"yahoo:^GSPC", "yahoo:HYG", "yahoo:IEF", "yahoo:DX-Y.NYB", "cboe:VIX",
              "yahoo:BZ=F", "fred:DGS10", "french"}       # inputs behind MARKET, DRIVERS and FACTOR
    hit = [f for f in failures if f in needed]
    if hit:
        raise SystemExit(f"❌ market inputs for the regressions failed to download: {hit}")
    return market


def build_market_sensitivities(tickers, market: pd.DataFrame, start: str = "2018-01-01",
                               window: int = 252, min_obs: int = 189) -> pd.DataFrame:
    """Long panel with columns: date, ticker, <FEATURES>."""
    if (market.index <= pd.Timestamp(start)).sum() < window:
        print(f"⚠️ market data has fewer than {window} days before {start}; early rows will be NaN")

    symbols = sorted({normalize_ticker(t) for t in tickers})
    px = yf.download(symbols, start=market.index[0], end=market.index[-1] + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False, threads=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(symbols[0])
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize().as_unit("ns")
    # No price filling: a missing close leaves the returns on both sides of it NaN
    returns = np.log(px.reindex(market.index)).diff()

    frames = []
    for i, sym in enumerate(symbols, 1):
        if sym not in returns or returns[sym].notna().sum() < min_obs:
            print(f"❌ {sym}: fewer than {min_obs} daily returns")
            continue
        df = sensitivities_for_returns(returns[sym], market, window, min_obs)
        df = df[df.index >= pd.Timestamp(start)].dropna(how="all")
        frames.append(df.assign(ticker=sym).rename_axis("date").reset_index())
        if i % 50 == 0 or i == len(symbols):
            print(f"✅ [{i}/{len(symbols)}] done")
    if not frames:
        return pd.DataFrame(columns=["date", "ticker"] + FEATURES)
    return pd.concat(frames, ignore_index=True)[["date", "ticker"] + FEATURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--etf", help="use all holdings from src/holdings_data/<ETF>-holdings.csv")
    ap.add_argument("--market", help="existing market_indicators.csv (default: build it in-process)")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--window", type=int, default=252, help="trailing trading days per regression")
    ap.add_argument("--min-obs", type=int, default=189, help="complete rows required in a window")
    ap.add_argument("--out", default="data/features/market_sensitivities.csv")
    args = ap.parse_args()

    tickers = args.tickers
    if args.etf:
        tickers = pd.read_csv(f"src/holdings_data/{args.etf}-holdings.csv")["Symbol"].dropna().tolist()
        tickers = [s for s in tickers if s != "--"]
    if not tickers:
        ap.error("pass --tickers or --etf")

    market = load_market(args.market, args.start)
    panel = build_market_sensitivities(tickers, market, args.start, args.window, args.min_obs)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    panel.to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, args.out)
    print(f"\nsaved {panel.shape} -> {args.out}")
    print(panel.groupby("ticker").tail(1).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
