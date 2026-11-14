import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))
import market_indicators as mi                              # noqa: E402
import market_sensitivities as ms                           # noqa: E402

D = pd.to_datetime


def _raw(n=400, seed=0) -> pd.DataFrame:
    """Synthetic raw inputs on a business-day calendar, every RAW_COLUMN filled."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n).as_unit("ns")
    raw = pd.DataFrame(index=idx)
    for col in mi.RAW_COLUMNS:
        raw[col] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return raw


# ---------------------------------------------------------------- point-in-time

def test_weekly_value_appears_only_after_release():
    nfci = pd.Series([0.5], index=D(["2024-01-05"]))         # Friday
    known = mi.to_known_dates(nfci, 4)
    cal = pd.bdate_range("2024-01-05", "2024-01-12").as_unit("ns")
    out = mi.asof_join(cal, known, 10)
    assert out[:"2024-01-10"].isna().all()                    # not visible through Wednesday
    assert out["2024-01-11":].eq(0.5).all()


def test_release_lag_keeps_latest_value_when_dates_collide():
    dff = pd.Series([5.0, 5.1, 5.2], index=D(["2024-01-05", "2024-01-06", "2024-01-07"]))   # Fri, Sat, Sun
    known = mi.to_known_dates(dff, 1)
    assert known.loc["2024-01-08"] == 5.2 and not known.index.duplicated().any()


def test_open_session_bar_is_dropped():
    s = pd.Series([1.0, 2.0], index=D(["2024-01-02", "2024-01-03"]))
    assert mi._clean(s, pd.Timestamp("2024-01-03")).index.tolist() == [pd.Timestamp("2024-01-02")]


def test_levels_expire_after_tolerance_and_returns_are_never_repeated():
    cal = pd.bdate_range("2024-01-01", "2024-01-12").as_unit("ns")
    level = pd.Series([4.0], index=D(["2024-01-01"]))
    assert mi.asof_join(cal, level, 5)["2024-01-05"] == 4.0
    assert np.isnan(mi.asof_join(cal, level, 5)["2024-01-08"])
    ret = pd.Series([0.01], index=D(["2024-01-01"]))
    assert mi.asof_join(cal, ret, 0).notna().sum() == 1


def test_market_features_do_not_look_ahead():
    raw = _raw()
    cut = raw.index[300]
    base = mi.build_market_features(raw)
    shocked = raw.copy()
    shocked.loc[shocked.index > cut] *= 3
    after = mi.build_market_features(shocked)
    pd.testing.assert_frame_equal(base.loc[:cut], after.loc[:cut])


# ---------------------------------------------------------------- feature definitions

def test_feature_columns_match_schema_even_when_sources_are_missing():
    out = mi.build_market_features(pd.DataFrame(index=_raw().index, columns=mi.RAW_COLUMNS, dtype=float))
    assert list(out.columns) == mi.FEATURES and out.isna().all(axis=None)


def test_sector_return_is_relative_to_spy():
    raw = _raw(n=30)
    raw["spy"] = 100.0
    raw["xlk"] = 100.0
    raw.loc[raw.index[-1], "xlk"] = 110.0
    out = mi.build_market_features(raw)
    assert out["xlk_rel_21d"].iloc[-1] == pytest.approx(np.log(1.1))


def test_curve_slope_and_vix_term_structure():
    raw = _raw(n=5)
    raw[["rate_10y", "rate_2y", "rate_3m", "vix", "vix3m"]] = [4.0, 4.5, 5.0, 30.0, 25.0]
    out = mi.build_market_features(raw).iloc[-1]
    assert out.slope_10y2y == pytest.approx(-0.5) and out.slope_10y3m == pytest.approx(-1.0)
    assert out.vix_term == pytest.approx(1.2)                 # inverted: near-term fear above 3-month


def test_french_parser_handles_header_footer_and_missing_codes():
    text = ("This file was created by CRSP.\r\n\r\n,Mkt-RF,SMB\r\n"
            "20240102,   1.00,  -0.50\r\n20240103, -99.99,   0.25\r\n\r\nCopyright 2026\r\n")
    df = mi.parse_french_csv(text)
    assert df.index.tolist() == list(D(["2024-01-02", "2024-01-03"]))
    assert df.loc["2024-01-02", "Mkt-RF"] == pytest.approx(0.01)
    assert np.isnan(df.loc["2024-01-03", "Mkt-RF"])


# ---------------------------------------------------------------- sensitivities

def test_rolling_ols_recovers_known_betas():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.01, (500, 2))
    y = 0.001 + 1.3 * X[:, 0] - 0.6 * X[:, 1] + rng.normal(0, 1e-4, 500)
    coefs, resid_std, r2 = ms.rolling_ols(y, X, window=252, min_obs=200)
    assert np.isnan(coefs[:251]).all()
    np.testing.assert_allclose(coefs[-1], [1.3, -0.6], atol=0.01)
    assert r2[-1] > 0.99 and resid_std[-1] == pytest.approx(1e-4, rel=0.15)


def test_rolling_ols_has_no_lookahead_and_needs_min_obs():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 0.01, (400, 1))
    y = 0.8 * X[:, 0] + rng.normal(0, 0.005, 400)
    y[:100] = np.nan                                          # e.g. IPO on day 100
    base, _, _ = ms.rolling_ols(y, X, window=100, min_obs=80)
    assert np.isnan(base[:179]).all() and np.isfinite(base[179:]).all()
    y2 = y.copy()
    y2[300:] *= -5
    after, _, _ = ms.rolling_ols(y2, X, window=100, min_obs=80)
    np.testing.assert_array_equal(base[:300], after[:300])


def test_driver_beta_is_exposure_beyond_the_market():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=400).as_unit("ns")
    spx = rng.normal(0, 0.01, 400)
    vix = -150 * spx + rng.normal(0, 0.5, 400)                # VIX mostly mirrors the market
    market = pd.DataFrame(0.0, index=idx, columns=[ms.MARKET] + list(ms.DRIVERS.values())
                          + list(ms.FACTOR.values()) + ["ff_rf"])
    market[ms.MARKET] = spx
    market["vix_chg_1d"] = vix
    market[list(ms.DRIVERS.values())[1:]] = rng.normal(0, 0.01, (400, 4))
    r = pd.Series(1.1 * spx + rng.normal(0, 1e-3, 400), index=idx)   # pure market stock
    out = ms.sensitivities_for_returns(r, market, window=252, min_obs=200).iloc[-1]
    assert out.beta_mkt == pytest.approx(1.1, abs=0.02)
    assert out.beta_vix == pytest.approx(0.0, abs=2e-4)


def test_oil_return_uses_only_same_day_and_prior_closes():
    # a shock to oil on day t must change oil_ret_1d on day t and no earlier day (no look-ahead)
    raw = _raw()
    base = mi.build_market_features(raw)
    shocked = raw.copy()
    t = 300
    shocked.iloc[t:, shocked.columns.get_loc("oil")] *= 1.2
    after = mi.build_market_features(shocked)
    np.testing.assert_array_equal(base.oil_ret_1d.iloc[:t], after.oil_ret_1d.iloc[:t])
    assert after.oil_ret_1d.iloc[t] == pytest.approx(base.oil_ret_1d.iloc[t] + np.log(1.2))


def test_beta_oil_pairs_same_day_returns():
    # stock = 0.7 * same-day oil move; a lagged oil input would give a beta near 0
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2020-01-01", periods=400).as_unit("ns")
    market = pd.DataFrame(0.0, index=idx, columns=[ms.MARKET] + list(ms.DRIVERS.values())
                          + list(ms.FACTOR.values()) + ["ff_rf"])
    market[ms.MARKET] = rng.normal(0, 0.01, 400)
    market["oil_ret_1d"] = rng.normal(0, 0.02, 400)
    r = pd.Series(0.7 * market["oil_ret_1d"] + rng.normal(0, 1e-3, 400), index=idx)
    out = ms.sensitivities_for_returns(r, market, window=252, min_obs=200).iloc[-1]
    assert out.beta_oil == pytest.approx(0.7, abs=0.03)
    assert mi.FRED["rate_10y"][1] == 0 and "oil" in mi.YAHOO and "oil" not in mi.FRED


def test_ticker_normalization():
    assert {ms.normalize_ticker(t) for t in ["BRK/B", "brk.b", " BRK-B "]} == {"BRK-B"}
