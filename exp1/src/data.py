"""Data loading: Yahoo Finance (run on your own machine), cached CSV, or synthetic.

Ingestion rules (none of these change the frozen config in config.py):
  * every ticker is downloaded on its own (threads=False, auto_adjust=True, timeout=30)
    with bounded retries and exponential backoff;
  * every ticker must have non-empty, finite, positive adjusted closes that cover the
    configured date range, otherwise nothing is written;
  * prices_daily.csv is only replaced atomically, after all tickers pass validation;
  * an existing cache with missing / empty / truncated ticker columns is detected and
    only the bad tickers are re-downloaded (no asset is ever dropped silently);
  * prices are never forward-filled: the weekly price is the last *observed* close in
    each W-FRI week, and a week where some (but not all) tickers have no observation
    is an error, not something to paper over.

Validate the data without running the experiment:
    python data.py              # check the cache, recover bad tickers if needed
    python data.py --download   # fresh download of all tickers, then validate
"""
from __future__ import annotations
import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config import Config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- download behaviour (operational only; does not affect results)
YF_TIMEOUT = 30            # seconds per request
MAX_ATTEMPTS = 5           # per ticker
BACKOFF_BASE = 2.0         # sleep 2, 4, 8, 16 s between attempts
PAUSE_BETWEEN_TICKERS = 1.0

# --- validation tolerances (reject data, never alter it)
START_TOLERANCE_DAYS = 10  # first close may be at most this many days after cfg.start
END_TOLERANCE_DAYS = 7     # last close may be at most this many days before cfg.end


class DataError(RuntimeError):
    """Raised when market data is missing, incomplete or inconsistent."""


# ---------------------------------------------------------------- extraction / validation

def extract_close(raw: pd.DataFrame | None, ticker: str) -> pd.Series:
    """Adjusted close of one ticker from a yfinance result.

    Handles flat columns ('Close'), MultiIndex (Price, Ticker) as returned by
    yfinance >= 0.2.48 by default, and (Ticker, Price) from group_by='ticker'.
    With auto_adjust=True, 'Close' is the adjusted close. Returns an empty series
    if the ticker or the Close field is absent.
    """
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float, name=ticker)
    cols = raw.columns
    s = None
    if isinstance(cols, pd.MultiIndex):
        for lvl in range(cols.nlevels):
            if "Close" in cols.get_level_values(lvl):
                sub = raw.xs("Close", axis=1, level=lvl)
                if isinstance(sub, pd.Series):
                    s = sub
                elif ticker in sub.columns:
                    s = sub[ticker]
                break                                 # other tickers only -> empty
    elif "Close" in cols:
        s = raw["Close"]
    if s is None:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(s, pd.DataFrame):          # duplicated column labels
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").astype(float)
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = ticker
    return s


def ticker_problems(s: pd.Series | None, ticker: str, cfg: Config) -> list[str]:
    """Reasons why a ticker's price series is unusable (empty list = OK)."""
    if s is None:
        return [f"{ticker}: column missing"]
    v = s.dropna()
    if v.empty:
        return [f"{ticker}: no prices (empty or all-NaN column)"]
    out = []
    if not np.isfinite(v.values).all() or (v.values <= 0).any():
        out.append(f"{ticker}: non-finite or non-positive prices")
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    if v.index[0] > start + pd.Timedelta(days=START_TOLERANCE_DAYS):
        out.append(f"{ticker}: history starts {v.index[0].date()}, expected near {start.date()}")
    if v.index[-1] < end - pd.Timedelta(days=END_TOLERANCE_DAYS):
        out.append(f"{ticker}: history ends {v.index[-1].date()}, expected near {end.date()}")
    if v.index[-1] > end:
        out.append(f"{ticker}: prices after configured end {end.date()}")
    return out


def audit_prices(px: pd.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Map ticker -> problems, for every configured ticker that is not usable."""
    bad = {}
    for t in cfg.tickers:
        p = ticker_problems(px[t] if t in px.columns else None, t, cfg)
        if p:
            bad[t] = p
    return bad


def panel_report(px: pd.DataFrame, cfg: Config) -> dict:
    """Common observation window and missing-observation counts (daily prices).

    Raises DataError if any ticker is unusable or the common window would discard
    a large part of the configured date range.
    """
    bad = audit_prices(px, cfg)
    if bad:
        raise DataError("invalid price data:\n  " + "\n  ".join(sum(bad.values(), [])))
    px = px[list(cfg.tickers)]
    first = {t: px[t].first_valid_index() for t in cfg.tickers}
    last = {t: px[t].last_valid_index() for t in cfg.tickers}
    c0, c1 = max(first.values()), min(last.values())
    if c0 >= c1:
        raise DataError(f"no common observation window (latest start {c0.date()}, "
                        f"earliest end {c1.date()})")
    win = px.loc[c0:c1]
    win = win[win.notna().any(axis=1)]                 # days on which the market traded
    missing = win.isna().sum()
    max_gap = {}
    for t in cfg.tickers:
        isna = win[t].isna().to_numpy()
        run = best = 0
        for x in isna:
            run = run + 1 if x else 0
            best = max(best, run)
        max_gap[t] = best
    return {
        "common_first": str(c0.date()), "common_last": str(c1.date()),
        "trading_days": int(len(win)),
        "rows_outside_common_window": int(px.notna().any(axis=1).sum() - len(win)),
        "missing_days_in_window": {t: int(missing[t]) for t in cfg.tickers},
        "max_consecutive_missing_days": max_gap,
        "per_ticker": {t: {"first": str(first[t].date()), "last": str(last[t].date()),
                           "obs": int(px[t].notna().sum())} for t in cfg.tickers},
    }


# ---------------------------------------------------------------- download

def download_ticker(ticker: str, cfg: Config, downloader: Callable | None = None,
                    max_attempts: int = MAX_ATTEMPTS, backoff: float = BACKOFF_BASE,
                    sleep: Callable[[float], None] = time.sleep) -> pd.Series:
    """Download one ticker's adjusted close with retries; raise DataError if it never
    yields a valid series."""
    if downloader is None:
        import yfinance as yf
        downloader = yf.download
    errors = []
    for attempt in range(1, max_attempts + 1):
        try:
            raw = downloader(ticker, start=cfg.start, end=cfg.end, auto_adjust=True,
                             threads=False, timeout=YF_TIMEOUT, progress=False)
            s = extract_close(raw, ticker)
            prob = ticker_problems(s, ticker, cfg)
            if not prob:
                return s
            errors.append(f"attempt {attempt}: " + "; ".join(prob))
        except Exception as e:                       # network / rate-limit / parse errors
            errors.append(f"attempt {attempt}: {type(e).__name__}: {e}")
        if attempt < max_attempts:
            wait = backoff * 2 ** (attempt - 1)
            print(f"  {ticker}: {errors[-1]} -> retry in {wait:g}s", flush=True)
            sleep(wait)
    raise DataError(f"{ticker}: download failed after {max_attempts} attempts:\n    "
                    + "\n    ".join(errors))


def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)                      # mkstemp defaults to 0600
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def download_prices(cfg: Config, path: Path | None = None, tickers=None,
                    existing: pd.DataFrame | None = None, **dl) -> pd.DataFrame:
    """Download `tickers` (default: all), merge with the valid columns of `existing`,
    validate all configured tickers and atomically write CSV + metadata.

    Nothing is written unless every ticker passes validation.
    """
    path = path or DATA_DIR / "prices_daily.csv"
    tickers = list(cfg.tickers if tickers is None else tickers)
    sleep = dl.get("sleep", time.sleep)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    fresh, failures = {}, []
    for k, t in enumerate(tickers):
        if k:
            sleep(PAUSE_BETWEEN_TICKERS)
        try:
            fresh[t] = download_ticker(t, cfg, **dl)
            print(f"  {t}: {fresh[t].notna().sum()} closes "
                  f"{fresh[t].index[0].date()} .. {fresh[t].index[-1].date()}", flush=True)
        except DataError as e:
            failures.append(str(e))
    if failures:
        raise DataError("not writing prices: some tickers failed\n  " + "\n  ".join(failures))

    cols = {}
    for t in cfg.tickers:
        if t in fresh:
            cols[t] = fresh[t]
        elif existing is not None and t in existing.columns:
            cols[t] = existing[t]
        else:
            raise DataError(f"{t}: neither downloaded nor present in existing cache")
    px = pd.DataFrame(cols)[list(cfg.tickers)].sort_index()
    px.index.name = "Date"
    px = px[px.notna().any(axis=1)]
    report = panel_report(px, cfg)                    # raises if anything is wrong

    old_meta = {}
    mp = _meta_path(path)
    if existing is not None and mp.exists():
        try:
            old_meta = json.loads(mp.read_text())
        except Exception:
            old_meta = {}
    old_dl = old_meta.get("ticker_downloaded_utc", {})
    try:
        import yfinance as yf
        yfv = getattr(yf, "__version__", "?")
    except ImportError:
        yfv = "?"
    meta = {"source": "Yahoo Finance via yfinance (auto_adjust=True, per-ticker download)",
            "downloaded_utc": now,
            "yfinance_version": yfv,
            "tickers": list(cfg.tickers),
            "ticker_downloaded_utc": {t: (now if t in fresh else
                                          old_dl.get(t, old_meta.get("downloaded_utc", "?")))
                                      for t in cfg.tickers},
            "recovered_tickers": list(tickers) if existing is not None else [],
            "rows": int(len(px)), "first": str(px.index[0].date()),
            "last": str(px.index[-1].date()),
            "validation": report}
    _atomic_write_text(path, px.to_csv())
    _atomic_write_text(mp, json.dumps(meta, indent=2))
    return px


def load_prices(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "prices_daily.csv"
    px = pd.read_csv(path, index_col=0, parse_dates=True)
    return px.sort_index()


def ensure_prices(cfg: Config, path: Path | None = None, refresh: bool = False,
                  **dl) -> pd.DataFrame:
    """Return validated daily prices for all configured tickers.

    refresh=True downloads everything. Otherwise the cache is audited and only
    missing / empty / truncated tickers are re-downloaded. The cache file is only
    replaced after the merged panel passes validation.
    """
    path = path or DATA_DIR / "prices_daily.csv"
    if refresh or not path.exists():
        print(f"downloading {len(cfg.tickers)} tickers -> {path}", flush=True)
        return download_prices(cfg, path, **dl)
    try:
        px = load_prices(path)
    except Exception as e:
        print(f"cache unreadable ({type(e).__name__}: {e}); downloading all tickers", flush=True)
        return download_prices(cfg, path, **dl)
    bad = audit_prices(px, cfg)
    if not bad:
        return px[list(cfg.tickers)]
    print("cache problems:\n  " + "\n  ".join(sum(bad.values(), [])), flush=True)
    print(f"recovering {list(bad)}", flush=True)
    return download_prices(cfg, path, tickers=list(bad), existing=px, **dl)


# ---------------------------------------------------------------- returns

def weekly_log_returns(px: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Last observed close of each W-FRI week -> weekly log returns.

    Restricted to the common observation window of all tickers (weeks before every
    asset has a price are dropped, as before). No forward- or back-filling: within a
    week the last *observed* close is used; a week in which some tickers have no
    observation at all raises DataError. Weeks with no observation for any ticker
    (market closed all week) are dropped and reported.
    """
    missing_cols = [t for t in cfg.tickers if t not in px.columns]
    if missing_cols:
        raise DataError(f"price data lacks tickers {missing_cols}")
    px = px[list(cfg.tickers)]
    empty = [t for t in cfg.tickers if px[t].notna().sum() == 0]
    if empty:
        raise DataError(f"tickers with no prices at all: {empty}")
    c0 = max(px[t].first_valid_index() for t in cfg.tickers)
    c1 = min(px[t].last_valid_index() for t in cfg.tickers)
    if c0 >= c1:
        raise DataError(f"no common observation window (latest start {c0.date()}, "
                        f"earliest end {c1.date()})")
    px = px.loc[c0:c1]

    wk = px.resample(cfg.week_rule).last()           # last non-NaN close per week
    closed = wk.isna().all(axis=1)
    if closed.any():
        print(f"  note: {int(closed.sum())} week(s) without any trading dropped: "
              f"{[str(d.date()) for d in wk.index[closed]]}", flush=True)
        wk = wk[~closed]
    holes = wk.isna().any(axis=1)
    if holes.any():
        det = {str(d.date()): [t for t in cfg.tickers if pd.isna(wk.at[d, t])]
               for d in wk.index[holes][:20]}
        raise DataError(f"{int(holes.sum())} week(s) where some tickers have no observed "
                        f"close (refusing to fill): {det}")

    r = np.log(wk).diff().iloc[1:]
    r = r.loc[: cfg.end]
    need = cfg.train_window + 1                      # >= one decision with r[i+1] observed
    if r.empty:
        raise DataError("weekly return panel is empty after alignment "
                        f"(common window {c0.date()} .. {c1.date()})")
    if len(r) < need:
        raise DataError(f"only {len(r)} weekly returns ({r.index[0].date()} .. "
                        f"{r.index[-1].date()}); need at least {need} "
                        f"(train_window={cfg.train_window} + 1)")
    if not np.isfinite(r.values).all():
        raise DataError("non-finite weekly returns")
    return r


def synthetic_returns(cfg: Config, n_weeks: int = 1450, seed: int = 0) -> pd.DataFrame:
    """Correlated AR(1)-GARCH(1,1)-t returns, only for testing the pipeline.

    Never report numbers from this data as results.
    """
    rng = np.random.default_rng(seed)
    d = len(cfg.tickers)
    A = rng.normal(size=(d, d))
    Cov = A @ A.T + d * np.eye(d)
    s = np.sqrt(np.diag(Cov))
    C = Cov / np.outer(s, s)
    L = np.linalg.cholesky(C)
    nu = 6.0
    omega, a, b = 0.02e-3, 0.08, 0.90
    c, phi = 0.0015, -0.05
    sig2 = np.full(d, omega / (1 - a - b))
    r_prev = np.zeros(d)
    eps_prev = np.zeros(d)
    out = np.empty((n_weeks, d))
    for t in range(n_weeks):
        sig2 = omega + a * eps_prev**2 + b * sig2
        g = rng.chisquare(nu) / nu
        z = (L @ rng.normal(size=d)) / np.sqrt(g) * np.sqrt((nu - 2) / nu)
        eps = np.sqrt(sig2) * z
        r = c + phi * r_prev + eps
        out[t] = r
        r_prev, eps_prev = r, eps
    idx = pd.date_range("1999-01-08", periods=n_weeks, freq=cfg.week_rule)
    return pd.DataFrame(out, index=idx, columns=list(cfg.tickers))


if __name__ == "__main__":
    import argparse
    from config import CFG
    ap = argparse.ArgumentParser(description="download / validate Exp1 price data")
    ap.add_argument("--download", action="store_true", help="fresh download of all tickers")
    a = ap.parse_args()
    px = ensure_prices(CFG, refresh=a.download)
    rep = panel_report(px, CFG)
    R = weekly_log_returns(px, CFG)
    print(json.dumps(rep, indent=2))
    print(f"daily prices: {px.shape}  {px.index[0].date()} .. {px.index[-1].date()}")
    print(f"weekly returns: {R.shape}  {R.index[0].date()} .. {R.index[-1].date()}  "
          f"NaN={int(R.isna().sum().sum())}")
    print("DATA OK")
