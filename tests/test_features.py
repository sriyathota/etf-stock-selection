import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))
from align import align_features, normalize_ticker          # noqa: E402
import financial_indicators as fi                           # noqa: E402

D = pd.to_datetime


def test_undefined_values_are_not_forward_filled():
    fin = pd.DataFrame({"date": D(["2024-01-02", "2024-01-03"]), "ticker": "INTC",
                        "pe_ratio": [12.0, np.nan]})          # turned loss-making on day 2
    out = align_features([fin])
    assert np.isnan(out.loc[out.date == "2024-01-03", "pe_ratio"].item())


def test_tickers_match_across_naming_styles():
    fin = pd.DataFrame({"date": D(["2024-01-02"]), "ticker": "BRK/B", "pe_ratio": [9.0]})
    px = pd.DataFrame({"date": D(["2024-01-02"]), "ticker": "BRK-B", "mom_1m": [0.1]})
    out = align_features([fin, px])
    assert len(out) == 1 and out.ticker.item() == "BRK-B"
    assert out[["pe_ratio", "mom_1m"]].notna().all(axis=None)


def test_market_values_use_latest_known_date_only():
    stock = pd.DataFrame({"date": D(["2024-01-02", "2024-01-03", "2024-01-04"]),
                          "ticker": "AAPL", "x": [1.0, 2.0, 3.0]})
    mkt = pd.DataFrame({"date": D(["2023-12-29", "2024-01-03"]), "vix": [12.0, 14.0]})
    out = align_features([stock], [mkt])
    assert out.vix.tolist() == [12.0, 14.0, 14.0]             # never sees 14.0 before 01-03


def test_market_values_expire_after_tolerance():
    stock = pd.DataFrame({"date": D(["2024-03-01"]), "ticker": "AAPL", "x": [1.0]})
    mkt = pd.DataFrame({"date": D(["2024-01-02"]), "rate": [5.3]})
    assert np.isnan(align_features([stock], [mkt]).rate.item())


def test_duplicate_rows_rejected():
    df = pd.DataFrame({"date": D(["2024-01-02"] * 2), "ticker": "AAPL", "x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        align_features([df])


def test_normalize_ticker():
    assert normalize_ticker(" brk.b ") == "BRK-B"


def _fake_quarters(period_ends, eps):
    cols = D(period_ends)
    inc = pd.DataFrame([eps, [1.0] * len(eps), [10.0] * len(eps)], columns=cols,
                       index=["Diluted EPS", "Net Income", "Total Revenue"])
    bal = pd.DataFrame([[5.0] * len(eps), [1.0] * len(eps), [1.0] * len(eps)], columns=cols,
                       index=["Stockholders Equity", "Total Debt", "Ordinary Shares Number"])
    return SimpleNamespace(quarterly_income_stmt=inc, quarterly_balance_sheet=bal)


def test_ttm_requires_consecutive_quarters():
    ok = fi._quarterly_reports(_fake_quarters(
        ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"], [1.0, 1.0, 1.0, 1.0]))
    assert ok.eps.tolist() == [4.0]
    gap = fi._quarterly_reports(_fake_quarters(                # Jun 2024 quarter missing
        ["2024-03-31", "2024-09-30", "2024-12-31", "2025-03-31"], [1.0, 1.0, 1.0, 1.0]))
    assert gap.empty


def test_quarterly_report_visible_only_after_lag():
    rep = fi._quarterly_reports(_fake_quarters(
        ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"], [1.0, 1.0, 1.0, 1.0]))
    assert rep.available.item() == D("2024-12-31") + fi.QUARTERLY_LAG
