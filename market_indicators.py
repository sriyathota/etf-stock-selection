"""
Market indicator features (Group 2 - Feature Engineering)

Builds a daily, date-level panel of market-wide features (no ticker column; this is the
"market-level" input of the feature alignment step):
    volatility   vix, vix_chg_1d/5d, vix_pct_1y, vix_term, vix_9d_ratio, vvix, skew, move, move_chg_5d
    rates        rate_3m/2y/10y/30y, slope_10y2y, slope_10y3m, fed_funds, breakeven_10y, real_10y,
                 rate_10y_chg_1d/5d, rate_2y_chg_5d
    credit       baa_spread, baa_spread_chg_21d, credit_ret_1d/21d (HYG vs IEF), ig_ret_21d (LQD vs IEF),
                 nfci, stlfsi
    S&P 500      spx_ret_1d/5d/21d/63d, spx_vol_21d, spx_drawdown_1y, spx_ma200_gap,
                 ew_minus_cw_21d (RSP vs SPY breadth), small_minus_large_21d (IWM vs SPY)
    sectors      <etf>_rel_21d/63d for the 11 SPDR sector ETFs (log return minus SPY), sector_dispersion_21d
    factors      Fama-French 5 + momentum daily returns (ff_*) and their 21-day sums
    macro        epu, epu_30d (policy uncertainty), dollar_ret_1d/21d, gold_ret_21d, oil_ret_1d/21d

Sources (all free, no API key): Yahoo via yfinance, CBOE index CSVs, FRED CSV downloads,
Ken French Data Library.

Point-in-time rule: each row's date is the trading day on whose close the value was KNOWN.
Values computed from same-day closing prices and published that evening (CBOE, Treasury yields,
Yahoo closes) keep their own date. Values the publisher computes after the fact are shifted
forward by RELEASE_LAG business days (fed funds, Moody's BAA, EPU, NFCI, STLFSI). Rows dated
today or later are dropped because Yahoo returns the still-open session as a bar.

BZ=F is Yahoo's continuous front-month Brent future and is NOT roll-adjusted, so oil_ret_1d and
oil_ret_21d include a small jump on each contract-roll day (monthly). beta_oil is a 252-day window,
so one jump per month adds a little noise but no look-ahead.

Known leaks yfinance/FRED cannot avoid: NFCI and STLFSI are re-estimated every week and FRED
serves only the latest vintage (ALFRED has the vintages). Fama-French factors are treated as
known at the close (they are a function of that day's prices), but the file is published about
monthly, so the last ~month of ff_* features is NaN.

Not used, on purpose: FRED's ICE BofA OAS series (truncated to the last 3 years by licensing;
BAA10Y and HYG/IEF cover credit back past 2018), FRED SP500 (10 years only), CBOE put/call
ratio files (frozen at 2019-10-04), WTI (went negative in April 2020; Brent did not), FRED Brent
spot DCOILBRENTEU (EIA posts it days late, and a release lag would misalign same-day oil betas).

Usage (from repo root):
    python src/features/market_indicators.py
    python src/features/market_indicators.py --start 2015-01-01 --out data/features/market_indicators.csv
"""

import argparse
import io
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from itertools import takewhile

import numpy as np
import pandas as pd
import yfinance as yf

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]

# raw column -> Yahoo symbol (closes are split/dividend adjusted)
YAHOO = {"spx": "^GSPC", "spy": "SPY", "rsp": "RSP", "iwm": "IWM", "move": "^MOVE",
         "hyg": "HYG", "lqd": "LQD", "ief": "IEF", "dollar": "DX-Y.NYB", "gold": "GLD",
         "oil": "BZ=F",                  # Brent front-month future; settles before the equity close
         **{s.lower(): s for s in SECTOR_ETFS}}

CBOE = {"vix": "VIX", "vix9d": "VIX9D", "vix3m": "VIX3M", "vvix": "VVIX", "skew": "SKEW"}
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"

# raw column -> (FRED id, release lag in business days, as-of tolerance in calendar days)
# Weekly lags = weekday of release after the Friday the value is dated + 1 day of slack for holiday weeks.
FRED = {
    "rate_3m": ("DGS3MO", 0, 5),
    "rate_2y": ("DGS2", 0, 5),
    "rate_10y": ("DGS10", 0, 5),
    "rate_30y": ("DGS30", 0, 5),
    "breakeven_10y": ("T10YIE", 0, 5),
    "real_10y": ("DFII10", 0, 5),
    "fed_funds": ("DFF", 1, 5),          # NY Fed publishes the next morning
    "baa_spread": ("BAA10Y", 1, 5),      # Moody's index posted after the close
    "epu": ("USEPUINDXD", 1, 5),         # built from day-t newspapers, posted next day
    "nfci": ("NFCI", 4, 10),             # week ending Friday, released Wednesday
    "stlfsi": ("STLFSI4", 5, 10),        # week ending Friday, released Thursday
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}&cosd={}"

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FRENCH_FILES = {
    "F-F_Research_Data_5_Factors_2x3_daily": {"Mkt-RF": "ff_mkt_rf", "SMB": "ff_smb", "HML": "ff_hml",
                                              "RMW": "ff_rmw", "CMA": "ff_cma", "RF": "ff_rf"},
    "F-F_Momentum_Factor_daily": {"Mom": "ff_mom"},
}
FF_FACTORS = ["ff_mkt_rf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_mom"]

RAW_COLUMNS = list(YAHOO) + list(CBOE) + list(FRED) + ["epu_30d"] + FF_FACTORS + ["ff_rf"]

WARMUP = pd.Timedelta(days=450)          # covers the 252-day windows before --start

FEATURES = (
    ["vix", "vix_chg_1d", "vix_chg_5d", "vix_pct_1y", "vix_term", "vix_9d_ratio",
     "vvix", "skew", "move", "move_chg_5d"]
    + ["rate_3m", "rate_2y", "rate_10y", "rate_30y", "slope_10y2y", "slope_10y3m", "fed_funds",
       "breakeven_10y", "real_10y", "rate_10y_chg_1d", "rate_10y_chg_5d", "rate_2y_chg_5d"]
    + ["baa_spread", "baa_spread_chg_21d", "credit_ret_1d", "credit_ret_21d", "ig_ret_21d",
       "nfci", "stlfsi"]
    + ["spx_ret_1d", "spx_ret_5d", "spx_ret_21d", "spx_ret_63d", "spx_vol_21d", "spx_drawdown_1y",
       "spx_ma200_gap", "ew_minus_cw_21d", "small_minus_large_21d"]
    + [f"{s.lower()}_rel_{h}" for s in SECTOR_ETFS for h in ("21d", "63d")] + ["sector_dispersion_21d"]
    + FF_FACTORS + ["ff_rf"] + [f"{c}_21d" for c in FF_FACTORS]
    + ["epu", "epu_30d", "dollar_ret_1d", "dollar_ret_21d", "gold_ret_21d", "oil_ret_1d", "oil_ret_21d"]
)


# ---------------------------------------------------------------- fetching

def _http_get(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (StockSelection research)"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def _clean(obj, end: pd.Timestamp):
    """Tz-naive ns daily index, sorted, deduplicated, and strictly before `end`."""
    idx = pd.DatetimeIndex(obj.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    obj = obj.copy()
    obj.index = idx.normalize().as_unit("ns")
    obj = obj[~obj.index.duplicated(keep="last")].sort_index()
    return obj[obj.index < end]


def fetch_yahoo(start, end) -> pd.DataFrame:
    px = yf.download(list(YAHOO.values()), start=start, end=end, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    px = px.rename(columns={v: k for k, v in YAHOO.items()})
    return _clean(px, end)


def fetch_cboe(symbol: str, end) -> pd.Series:
    df = pd.read_csv(io.BytesIO(_http_get(CBOE_URL.format(symbol))))
    df.columns = df.columns.str.strip().str.upper()
    value = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]      # VVIX/SKEW files have one value column
    s = pd.Series(pd.to_numeric(df[value], errors="coerce").values,
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y"))
    return _clean(s, end).dropna()


def fetch_fred(series_id: str, start, end) -> pd.Series:
    df = pd.read_csv(io.BytesIO(_http_get(FRED_URL.format(series_id, pd.Timestamp(start).date()))),
                     na_values=["."])
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]))
    return _clean(s, end).dropna()


def parse_french_csv(text: str) -> pd.DataFrame:
    """First table of a Ken French daily CSV, in decimal returns, indexed by date."""
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith(","))
    rows = list(takewhile(lambda line: re.match(r"\s*\d{8}\s*,", line), lines[header + 1:]))
    df = pd.read_csv(io.StringIO("\n".join([lines[header]] + rows)), index_col=0)
    df.columns = df.columns.str.strip()
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    return df.replace([-99.99, -999], np.nan) / 100


def fetch_french(end) -> pd.DataFrame:
    frames = []
    for name, columns in FRENCH_FILES.items():
        with zipfile.ZipFile(io.BytesIO(_http_get(FRENCH_URL.format(name)))) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
        frames.append(parse_french_csv(text)[list(columns)].rename(columns=columns))
    return _clean(pd.concat(frames, axis=1, sort=True), end)


# ---------------------------------------------------------------- point-in-time alignment

def to_known_dates(s: pd.Series, lag_bdays: int) -> pd.Series:
    """Re-date each value to the business day it became public; the latest observation wins a tie."""
    if not lag_bdays:
        return s
    s = s.sort_index()
    s.index = s.index + pd.offsets.BDay(lag_bdays)
    return s[~s.index.duplicated(keep="last")]


def asof_join(calendar: pd.DatetimeIndex, s: pd.Series, tolerance_days: int) -> pd.Series:
    """Latest value dated on or before each calendar day, expiring after `tolerance_days`.
    tolerance_days=0 means exact-date match only (for daily returns, which must not be repeated)."""
    calendar = pd.DatetimeIndex(calendar).as_unit("ns")          # pandas 3 may infer s/us units
    s = s.set_axis(pd.DatetimeIndex(s.index).as_unit("ns"))
    if tolerance_days == 0:
        return s.reindex(calendar)
    left = pd.DataFrame(index=calendar)
    right = s.sort_index().rename("value").to_frame()
    out = pd.merge_asof(left, right, left_index=True, right_index=True, direction="backward",
                        tolerance=pd.Timedelta(days=tolerance_days))
    return out["value"]


def fetch_raw(start, end) -> tuple:
    """All raw inputs on the S&P 500 trading calendar. Returns (raw frame, list of failed sources)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    failures = []

    yahoo = fetch_yahoo(start, end)
    if "spx" not in yahoo or yahoo["spx"].dropna().empty:
        raise RuntimeError("could not download ^GSPC, which defines the trading calendar")
    calendar = yahoo["spx"].dropna().index
    raw = yahoo.reindex(calendar)
    missing = [YAHOO[c] for c in YAHOO if c not in raw or raw[c].dropna().empty]
    failures += [f"yahoo:{m}" for m in missing]
    print(("❌" if missing else "✅") + f" yahoo: {len(YAHOO) - len(missing)}/{len(YAHOO)} symbols"
          + (f" (missing {missing})" if missing else ""))

    for col, symbol in CBOE.items():
        try:
            raw[col] = asof_join(calendar, fetch_cboe(symbol, end), 5)
            print(f"✅ cboe:{symbol}")
        except Exception as e:
            failures.append(f"cboe:{symbol}")
            print(f"❌ cboe:{symbol}: {e}")

    for col, (series_id, lag, tolerance) in FRED.items():
        try:
            s = fetch_fred(series_id, start, end)
            if col == "epu":    # 30-day mean over the calendar-daily series, before re-dating
                raw["epu_30d"] = asof_join(calendar, to_known_dates(s.rolling("30D").mean(), lag), tolerance)
            raw[col] = asof_join(calendar, to_known_dates(s, lag), tolerance)
            print(f"✅ fred:{series_id}")
        except Exception as e:
            failures.append(f"fred:{series_id}")
            print(f"❌ fred:{series_id}: {e}")
        time.sleep(0.2)

    try:
        french = fetch_french(end)
        for col in french:
            raw[col] = asof_join(calendar, french[col], 0)
        print(f"✅ french: factors through {french.index.max().date()}")
    except Exception as e:
        failures.append("french")
        print(f"❌ french: {e}")

    return raw.reindex(columns=RAW_COLUMNS), failures


# ---------------------------------------------------------------- features

def _logret(px: pd.Series, n: int = 1) -> pd.Series:
    return np.log(px / px.shift(n))


def build_market_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Market features from raw inputs already aligned to the trading calendar (index = date).
    Every feature at row t uses only rows <= t."""
    f = pd.DataFrame(index=raw.index)

    vix = raw["vix"]
    f["vix"] = vix
    f["vix_chg_1d"] = vix.diff()
    f["vix_chg_5d"] = vix.diff(5)
    f["vix_pct_1y"] = vix.rolling(252, min_periods=126).rank(pct=True)
    f["vix_term"] = vix / raw["vix3m"]                   # > 1: inverted term structure (stress)
    f["vix_9d_ratio"] = raw["vix9d"] / vix
    f["vvix"] = raw["vvix"]
    f["skew"] = raw["skew"]
    f["move"] = raw["move"]
    f["move_chg_5d"] = raw["move"].diff(5)

    # Rates are in percentage points
    for col in ["rate_3m", "rate_2y", "rate_10y", "rate_30y"]:
        f[col] = raw[col]
    f["slope_10y2y"] = raw["rate_10y"] - raw["rate_2y"]
    f["slope_10y3m"] = raw["rate_10y"] - raw["rate_3m"]
    for col in ["fed_funds", "breakeven_10y", "real_10y"]:
        f[col] = raw[col]
    f["rate_10y_chg_1d"] = raw["rate_10y"].diff()
    f["rate_10y_chg_5d"] = raw["rate_10y"].diff(5)
    f["rate_2y_chg_5d"] = raw["rate_2y"].diff(5)

    f["baa_spread"] = raw["baa_spread"]
    f["baa_spread_chg_21d"] = raw["baa_spread"].diff(21)
    f["credit_ret_1d"] = _logret(raw["hyg"]) - _logret(raw["ief"])       # high yield minus Treasuries: credit risk appetite
    f["credit_ret_21d"] = _logret(raw["hyg"], 21) - _logret(raw["ief"], 21)
    f["ig_ret_21d"] = _logret(raw["lqd"], 21) - _logret(raw["ief"], 21)
    f["nfci"] = raw["nfci"]
    f["stlfsi"] = raw["stlfsi"]

    spx = raw["spx"]
    spx_1d = _logret(spx)
    f["spx_ret_1d"] = spx_1d
    for n in (5, 21, 63):
        f[f"spx_ret_{n}d"] = _logret(spx, n)
    f["spx_vol_21d"] = spx_1d.rolling(21).std() * np.sqrt(252)
    f["spx_drawdown_1y"] = spx / spx.rolling(252, min_periods=126).max() - 1
    f["spx_ma200_gap"] = spx / spx.rolling(200).mean() - 1
    spy_21d, spy_63d = _logret(raw["spy"], 21), _logret(raw["spy"], 63)
    f["ew_minus_cw_21d"] = _logret(raw["rsp"], 21) - spy_21d
    f["small_minus_large_21d"] = _logret(raw["iwm"], 21) - spy_21d

    sector_21d = pd.DataFrame({s: _logret(raw[s.lower()], 21) for s in SECTOR_ETFS})
    for s in SECTOR_ETFS:
        f[f"{s.lower()}_rel_21d"] = sector_21d[s] - spy_21d
        f[f"{s.lower()}_rel_63d"] = _logret(raw[s.lower()], 63) - spy_63d
    # XLRE (2015) and XLC (2018) are young; require most sectors rather than backfilling them
    f["sector_dispersion_21d"] = sector_21d.std(axis=1).where(sector_21d.notna().sum(axis=1) >= 9)

    for col in FF_FACTORS + ["ff_rf"]:
        f[col] = raw[col]
    for col in FF_FACTORS:
        f[f"{col}_21d"] = raw[col].rolling(21).sum()

    f["epu"] = raw["epu"]
    f["epu_30d"] = raw["epu_30d"]
    f["dollar_ret_1d"] = _logret(raw["dollar"])
    f["dollar_ret_21d"] = _logret(raw["dollar"], 21)
    f["gold_ret_21d"] = _logret(raw["gold"], 21)
    f["oil_ret_1d"] = _logret(raw["oil"])
    f["oil_ret_21d"] = _logret(raw["oil"], 21)

    return f[FEATURES].replace([np.inf, -np.inf], np.nan)


def build_market_indicators(start: str = "2018-01-01", end: str = None) -> tuple:
    """Feature frame (index = date, from `start`) plus the list of sources that failed."""
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()   # exclusive: drops today's open bar
    raw, failures = fetch_raw(pd.Timestamp(start) - WARMUP, end)
    features = build_market_features(raw)
    return features[features.index >= pd.Timestamp(start)], failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", help="exclusive; default today, so a still-open session is never used")
    ap.add_argument("--out", default="data/features/market_indicators.csv")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the file even if a source failed (its features stay NaN)")
    args = ap.parse_args()

    features, failures = build_market_indicators(args.start, args.end)
    if failures and not args.allow_partial:
        raise SystemExit(f"\n❌ {len(failures)} source(s) failed: {failures}. Nothing written; "
                         "re-run, or pass --allow-partial to keep their features NaN.")

    out = features.rename_axis("date").reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    out.to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, args.out)                             # never leave a half-written file behind
    print(f"\nsaved {out.shape} ({out.date.iloc[0]} -> {out.date.iloc[-1]}) -> {args.out}")
    print("non-null share per feature (lowest first):")
    print(features.notna().mean().sort_values().head(15).round(3).to_string())


if __name__ == "__main__":
    main()
