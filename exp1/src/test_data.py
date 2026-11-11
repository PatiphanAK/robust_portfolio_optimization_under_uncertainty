"""Data-ingestion tests (no network: yfinance is replaced by a fake downloader).

Run: python test_data.py
"""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG
import data as D

TICKERS = list(CFG.tickers)
NOSLEEP = {"sleep": lambda s: None}


def fake_prices(ticker: str, start=CFG.start, end=CFG.end) -> pd.Series:
    """Deterministic positive random-walk closes on business days in [start, end)."""
    idx = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
    rng = np.random.default_rng(sum(map(ord, ticker)))
    p = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    return pd.Series(p, index=idx, name=ticker)


def yf_frame(s: pd.Series, layout="price_ticker") -> pd.DataFrame:
    """Mimic yfinance >= 0.2.48 output for a single ticker."""
    t = s.name
    f = pd.DataFrame({"Close": s, "High": s * 1.01, "Low": s * 0.99, "Open": s,
                      "Volume": 1e6}, index=s.index)
    f.index.name = "Date"
    if layout == "flat":
        return f
    if layout == "price_ticker":
        f.columns = pd.MultiIndex.from_product([f.columns, [t]], names=["Price", "Ticker"])
    else:  # ticker_price (group_by='ticker')
        f.columns = pd.MultiIndex.from_product([[t], f.columns], names=["Ticker", "Price"])
    return f


class FakeYF:
    """Fake yf.download. `plan[ticker]` is a list of behaviours consumed per call:
    'ok', 'empty', 'raise', 'truncated', 'nan'. Default is 'ok' forever."""

    def __init__(self, plan=None):
        self.plan = {k: list(v) for k, v in (plan or {}).items()}
        self.calls = []

    def __call__(self, ticker, **kw):
        self.calls.append((ticker, kw))
        q = self.plan.get(ticker)
        mode = (q.pop(0) if len(q) > 1 else q[0]) if q else "ok"
        if mode == "raise":
            raise ConnectionError("simulated rate limit")
        if mode == "empty":
            return pd.DataFrame()
        s = fake_prices(ticker, kw["start"], kw["end"])
        if mode == "truncated":
            s = s.loc[:"2010-06-30"]
        if mode == "nan":
            s[:] = np.nan
        return yf_frame(s)

    def tickers_called(self):
        return [c[0] for c in self.calls]


def full_panel():
    return pd.DataFrame({t: fake_prices(t) for t in TICKERS})


class TestExtract(unittest.TestCase):
    def test_multiindex_price_ticker(self):
        s = fake_prices("XLE")
        out = D.extract_close(yf_frame(s, "price_ticker"), "XLE")
        pd.testing.assert_series_equal(out, s, check_names=False, check_freq=False)

    def test_multiindex_ticker_price(self):
        s = fake_prices("XLF")
        out = D.extract_close(yf_frame(s, "ticker_price"), "XLF")
        pd.testing.assert_series_equal(out, s, check_names=False, check_freq=False)

    def test_flat_columns(self):
        s = fake_prices("XLI")
        out = D.extract_close(yf_frame(s, "flat"), "XLI")
        np.testing.assert_allclose(out.values, s.values)

    def test_wrong_ticker_is_empty(self):
        out = D.extract_close(yf_frame(fake_prices("XLK")), "XLE")
        self.assertTrue(out.empty)

    def test_empty_and_none(self):
        self.assertTrue(D.extract_close(pd.DataFrame(), "XLE").empty)
        self.assertTrue(D.extract_close(None, "XLE").empty)

    def test_tz_aware_index_is_normalised(self):
        s = fake_prices("XLP")
        f = yf_frame(s)
        f.index = f.index.tz_localize("America/New_York")
        out = D.extract_close(f, "XLP")
        self.assertIsNone(out.index.tz)
        self.assertTrue((out.index == s.index).all())


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "prices_daily.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_writes_all_nine(self):
        fy = FakeYF()
        px = D.download_prices(CFG, self.path, downloader=fy, **NOSLEEP)
        self.assertEqual(fy.tickers_called(), TICKERS)              # one call per ticker
        for _, kw in fy.calls:
            self.assertEqual((kw["threads"], kw["auto_adjust"], kw["timeout"]), (False, True, 30))
            self.assertEqual((kw["start"], kw["end"]), (CFG.start, CFG.end))
        self.assertEqual(list(px.columns), TICKERS)
        disk = D.load_prices(self.path)
        self.assertEqual(list(disk.columns), TICKERS)
        self.assertEqual(int(disk.isna().sum().sum()), 0)
        self.assertTrue(self.path.with_suffix(".meta.json").exists())
        self.assertEqual(list(Path(self.tmp.name).glob("*.tmp")), [])  # no temp leftovers

    def test_empty_ticker_retries_then_fails_without_writing(self):
        waits = []
        fy = FakeYF({"XLE": ["empty"]})
        with self.assertRaises(D.DataError) as cm:
            D.download_prices(CFG, self.path, downloader=fy, sleep=waits.append)
        self.assertIn("XLE", str(cm.exception))
        self.assertEqual(fy.tickers_called().count("XLE"), D.MAX_ATTEMPTS)
        backoff = [w for w in waits if w != D.PAUSE_BETWEEN_TICKERS]
        self.assertEqual(backoff, [D.BACKOFF_BASE * 2**k for k in range(D.MAX_ATTEMPTS - 1)])
        self.assertFalse(self.path.exists())

    def test_all_nan_ticker_rejected(self):
        with self.assertRaises(D.DataError):
            D.download_prices(CFG, self.path, downloader=FakeYF({"XLV": ["nan"]}), **NOSLEEP)
        self.assertFalse(self.path.exists())

    def test_partial_history_rejected(self):
        with self.assertRaises(D.DataError) as cm:
            D.download_prices(CFG, self.path, downloader=FakeYF({"XLY": ["truncated"]}),
                              **NOSLEEP)
        self.assertIn("history ends", str(cm.exception))
        self.assertFalse(self.path.exists())

    def test_transient_failures_recover(self):
        fy = FakeYF({"XLF": ["raise", "empty", "ok"]})
        px = D.download_prices(CFG, self.path, downloader=fy, **NOSLEEP)
        self.assertEqual(fy.tickers_called().count("XLF"), 3)
        self.assertEqual(int(px.isna().sum().sum()), 0)


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "prices_daily.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_corrupt(self, drop_col=None):
        px = full_panel()
        bad = ["XLE", "XLF", "XLI", "XLP", "XLV", "XLY"]       # the real failure pattern
        px[bad] = np.nan
        if drop_col:
            px = px.drop(columns=drop_col)
        px.index.name = "Date"
        px.to_csv(self.path)
        return px

    def test_audit_detects_empty_and_missing_columns(self):
        px = self._write_corrupt(drop_col="XLK")
        bad = D.audit_prices(D.load_prices(self.path), CFG)
        self.assertEqual(sorted(bad), sorted(["XLE", "XLF", "XLI", "XLP", "XLV", "XLY", "XLK"]))
        self.assertIn("column missing", bad["XLK"][0])

    def test_corrupt_cache_recovers_only_bad_tickers(self):
        before = self._write_corrupt()
        fy = FakeYF()
        px = D.ensure_prices(CFG, self.path, downloader=fy, **NOSLEEP)
        self.assertEqual(sorted(fy.tickers_called()),
                         sorted(["XLE", "XLF", "XLI", "XLP", "XLV", "XLY"]))
        disk = D.load_prices(self.path)
        self.assertEqual(list(disk.columns), TICKERS)
        self.assertEqual(int(disk.isna().sum().sum()), 0)
        for t in ["XLB", "XLK", "XLU"]:                         # good columns untouched
            np.testing.assert_allclose(disk[t].values, before[t].values, rtol=1e-12)
        self.assertEqual(D.audit_prices(px, CFG), {})
        R = D.weekly_log_returns(px, CFG)
        self.assertEqual(R.shape[1], 9)

    def test_failed_recovery_leaves_cache_unchanged(self):
        self._write_corrupt()
        raw = self.path.read_bytes()
        with self.assertRaises(D.DataError):
            D.ensure_prices(CFG, self.path, downloader=FakeYF({"XLP": ["raise"]}), **NOSLEEP)
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual(list(Path(self.tmp.name).glob("*.tmp")), [])

    def test_unreadable_cache_redownloads(self):
        self.path.write_text("\x00garbage")
        fy = FakeYF()
        D.ensure_prices(CFG, self.path, downloader=fy, **NOSLEEP)
        self.assertEqual(fy.tickers_called(), TICKERS)

    def test_valid_cache_not_redownloaded(self):
        full_panel().rename_axis("Date").to_csv(self.path)
        fy = FakeYF()
        px = D.ensure_prices(CFG, self.path, downloader=fy, **NOSLEEP)
        self.assertEqual(fy.calls, [])
        self.assertEqual(px.shape[1], 9)


class TestWeekly(unittest.TestCase):
    def test_complete_panel(self):
        px = full_panel()
        R = D.weekly_log_returns(px, CFG)
        self.assertEqual(list(R.columns), TICKERS)
        self.assertTrue((R.index.day_of_week == 4).all())          # W-FRI labels
        self.assertEqual(int(R.isna().sum().sum()), 0)
        wk = px.resample("W-FRI").last()
        np.testing.assert_allclose(R.values, np.log(wk).diff().iloc[1:].values)

    def test_single_missing_day_uses_last_observed_close(self):
        px = full_panel()
        fri = px.index[(px.index.day_of_week == 4)][100]
        px.loc[fri, "XLE"] = np.nan
        R = D.weekly_log_returns(px, CFG)
        thu = fri - pd.Timedelta(days=1)
        prev = px["XLE"].loc[:fri - pd.Timedelta(days=7)].iloc[-1]
        self.assertAlmostEqual(R.loc[fri, "XLE"], np.log(px.loc[thu, "XLE"] / prev))

    def test_whole_week_gap_raises(self):
        px = full_panel()
        px.loc["2005-03-07":"2005-03-11", "XLV"] = np.nan
        with self.assertRaises(D.DataError) as cm:
            D.weekly_log_returns(px, CFG)
        self.assertIn("2005-03-11", str(cm.exception))

    def test_market_closed_week_dropped(self):
        px = full_panel()
        px = px.drop(px.loc["2005-03-07":"2005-03-11"].index)
        R = D.weekly_log_returns(px, CFG)
        self.assertNotIn(pd.Timestamp("2005-03-11"), R.index)

    def test_empty_ticker_raises(self):
        px = full_panel()
        px["XLE"] = np.nan
        with self.assertRaises(D.DataError) as cm:
            D.weekly_log_returns(px, CFG)
        self.assertIn("XLE", str(cm.exception))

    def test_missing_column_raises(self):
        with self.assertRaises(D.DataError):
            D.weekly_log_returns(full_panel().drop(columns="XLY"), CFG)

    def test_too_short_raises(self):
        px = full_panel().loc["2020-01-01":"2022-01-01"]           # ~104 weeks < 261
        with self.assertRaises(D.DataError) as cm:
            D.weekly_log_returns(px, CFG)
        self.assertIn("need at least", str(cm.exception))

    def test_no_overlap_raises(self):
        px = full_panel()
        px.loc["2005":, "XLB"] = np.nan
        px.loc[:"2006", "XLK"] = np.nan
        with self.assertRaises(D.DataError):
            D.weekly_log_returns(px, CFG)

    def test_panel_report_counts_missing(self):
        px = full_panel()
        px.iloc[500:503, 2] = np.nan
        rep = D.panel_report(px, CFG)
        self.assertEqual(rep["missing_days_in_window"]["XLF"], 3)
        self.assertEqual(rep["max_consecutive_missing_days"]["XLF"], 3)
        self.assertEqual(sum(rep["missing_days_in_window"].values()), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
