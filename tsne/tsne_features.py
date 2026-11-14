import numpy as np
import pandas as pd

NUMERIC_STATIC = [
    "market_cap","pe_ratio","beta","dividend_yield","profit_margins",
    "revenue_growth","debt_to_equity","return_on_equity"
]

def _safe(series, fn, default=np.nan):
    try:
        v = fn(series.dropna())
        return default if (v is None or np.isinf(v) or pd.isna(v)) else v
    except Exception:
        return default

def build_feature_rows(data_dict, ref_benchmark_close=None, window=120):
    """
    Returns:
      X: DataFrame (n_stocks x n_features)
      meta: DataFrame with columns [ticker, etf, sector, industry, weight?]
    """
    rows, meta = [], []
    for etf, holdings in data_dict.items():
        for ticker, obj in holdings.items():
            dd = obj.get("daily_data", pd.DataFrame()).tail(window).copy()
            if dd.empty or "Close" not in dd.columns:
                continue

            # returns series
            ret = dd.get("Daily_Return", dd["Close"].pct_change()).dropna()
            close = dd["Close"]

            # basic moments / momentum / drawdown
            feats = dict(
                ret_mean=_safe(ret, np.mean),
                ret_std=_safe(ret, np.std),
                ret_skew=_safe(ret, pd.Series.skew),
                ret_kurt=_safe(ret, pd.Series.kurt),
                mom_1m=_safe(ret.tail(21), lambda s:(1+s).prod()-1, 0.0),
                mom_3m=_safe(ret.tail(63), lambda s:(1+s).prod()-1, 0.0),
                mom_6m=_safe(ret.tail(126), lambda s:(1+s).prod()-1, 0.0),
                max_dd=_safe(close/close.cummax()-1, np.min, 0.0),
                rsi=_safe(dd.get("RSI", pd.Series(dtype=float)), np.mean, 0.0),
                vol30=_safe(dd.get("Volatility_30d", pd.Series(dtype=float)), np.mean, 0.0),
                ma20_gap=_safe((dd.get("MA_20", close)-close)/close, np.mean, 0.0),
                ma50_gap=_safe((dd.get("MA_50", close)-close)/close, np.mean, 0.0),
                log_dollar_vol=_safe(close*dd.get("Volume", pd.Series(dtype=float)), lambda s: np.log1p(s.mean()), 0.0),
            )

            # add numeric static features
            sf = obj.get("static_features", {}) or {}
            for k in NUMERIC_STATIC:
                v = sf.get(k, np.nan)
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                feats[f"static_{k}"] = v

            rows.append(feats)
            meta.append({
                "ticker": ticker,
                "etf": etf,
                "sector": sf.get("sector", "Unknown"),
                "industry": sf.get("industry", "Unknown"),
            })

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    meta = pd.DataFrame(meta)
    # simple impute
    X = X.fillna(X.median(numeric_only=True))
    return X, meta
