"""Data loading: Yahoo Finance (run on your own machine), cached CSV, or synthetic."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_prices(cfg: Config, path: Path | None = None) -> pd.DataFrame:
    """Download daily adjusted close with yfinance and save to CSV + metadata."""
    import yfinance as yf

    path = path or DATA_DIR / "prices_daily.csv"
    raw = yf.download(list(cfg.tickers), start=cfg.start, end=cfg.end,
                      auto_adjust=True, progress=False)
    px = raw["Close"][list(cfg.tickers)]
    path.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(path)
    meta = {"source": "Yahoo Finance via yfinance (auto_adjust=True)",
            "downloaded_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "yfinance_version": getattr(yf, "__version__", "?"),
            "rows": int(len(px)), "first": str(px.index[0].date()), "last": str(px.index[-1].date())}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return px


def load_prices(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "prices_daily.csv"
    px = pd.read_csv(path, index_col=0, parse_dates=True)
    return px.sort_index()


def weekly_log_returns(px: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Last available price of each week -> weekly log returns.

    Missing days are forward-filled only (never back-filled) to avoid look-ahead.
    Weeks before every asset has a price are dropped, not filled.
    """
    px = px[list(cfg.tickers)].ffill()
    wk = px.resample(cfg.week_rule).last()
    r = np.log(wk).diff().dropna(how="any")
    return r.loc[: cfg.end]


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
