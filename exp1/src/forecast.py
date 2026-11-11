"""AR(1)-GARCH(1,1) with standardized Student-t innovations, per asset.

At decision index i (end of week i) only returns r[i-W+1 .. i] are used.
Parameters are re-estimated every `refit_every` decisions and held fixed in
between (the GARCH filter is still run on the newest data each week).
Output for each decision:
    mu    (d,)   one-step conditional mean of r[i+1]
    sigma (d,)   one-step conditional std of r[i+1]
    Z     (N,d)  joint standardized residuals of the window (rows = same week,
                 which keeps cross-asset dependence for filtered historical simulation)
    nu    (d,)   fitted degrees of freedom
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from config import Config


def _model(x: np.ndarray):
    return arch_model(x, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t", rescale=False)


def _fit(x: np.ndarray, prev_params=None):
    am = _model(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", starting_values=prev_params, options={"maxiter": 500})
    p = res.params.values
    ok = res.convergence_flag == 0 and np.all(np.isfinite(p)) and p[3] + p[4] < 0.9999
    if not ok and prev_params is not None:
        return prev_params, False
    return p, ok


def _filter(x: np.ndarray, params: np.ndarray):
    am = _model(x)
    fr = am.fix(params)
    f = fr.forecast(horizon=1, reindex=False)
    return float(f.mean.values[-1, 0]), float(f.variance.values[-1, 0]), np.asarray(fr.std_resid)


def run_forecasts(R: pd.DataFrame, cfg: Config, cache: Path | None = None, verbose=True):
    """Return list of dicts, one per decision index i (forecast for r[i+1])."""
    if cache is not None and cache.exists():
        return list(np.load(cache, allow_pickle=True)["out"])
    X = R.values * cfg.scale
    T, d = X.shape
    W = cfg.train_window
    params = [None] * d
    out = []
    n_fail = 0
    decisions = range(W - 1, T - 1)          # need r[i+1] to evaluate
    for k, i in enumerate(decisions):
        win = X[i - W + 1 : i + 1]
        assert win.shape[0] == W and i + 1 < T   # window ends at i: no look-ahead
        refit = k % cfg.refit_every == 0
        mu = np.empty(d); var = np.empty(d); nu = np.empty(d)
        Zcols = []
        for j in range(d):
            if refit or params[j] is None:
                params[j], ok = _fit(win[:, j], params[j])
                n_fail += (not ok)
            m, v, z = _filter(win[:, j], params[j])
            mu[j], var[j], nu[j] = m, v, params[j][-1]
            Zcols.append(z)
        Z = np.column_stack(Zcols)
        Z = Z[~np.isnan(Z).any(axis=1)]
        out.append({"i": i, "date": R.index[i], "mu": mu / cfg.scale,
                    "sigma": np.sqrt(var) / cfg.scale, "Z": Z, "nu": nu, "refit": refit})
        if verbose and k % 100 == 0:
            print(f"  forecast {k+1}/{len(decisions)}  {R.index[i].date()}  fails={n_fail}", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, out=np.array(out, dtype=object))
    return out
