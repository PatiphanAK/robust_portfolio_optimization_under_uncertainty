"""Rolling-origin backtest: every model sees exactly the same forecasts and scenarios."""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from ambiguity import eps_base, kappas_elliptical, possibility_cuts
from optimize import NominalWass, PossA, PossB, clean


def model_names(cfg: Config, appendix=True):
    scales = list(cfg.eps_scales) + (list(cfg.eps_scales_appendix) if appendix else [])
    return ["Nominal"] + [f"Wass x{s:g}" for s in sorted(scales)] + ["Poss-A", "Poss-B"]


def run_backtest(R: pd.DataFrame, fc: list, cfg: Config, cache: Path | None = None,
                 appendix=True, verbose=True):
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["W"].item(), z["info"].item()
    rng = np.random.default_rng(cfg.seed)
    lams = np.array(cfg.lambdas)
    names = model_names(cfg, appendix)
    scales = sorted(list(cfg.eps_scales) + (list(cfg.eps_scales_appendix) if appendix else []))
    N, d = fc[0]["Z"].shape
    K = len(cfg.poss_gammas)
    nw = NominalWass(N, d, cfg.alpha, cfg.ridge)
    pa = PossA(d, cfg.ridge)
    b = 1 - np.array(cfg.poss_gammas)
    pb = PossB(N, d, K, b, cfg.alpha, cfg.ridge)

    T = len(fc)
    W = {m: np.full((T, len(lams), d), np.nan) for m in names}
    eps0 = np.full(T, np.nan); kE = np.full(T, np.nan); kC = np.full(T, np.nan)
    bad = {m: 0 for m in names}          # failed solves -> previous weights reused
    inacc = {m: 0 for m in names}        # accepted but flagged optimal_inaccurate
    t0 = time.time()
    for k, f in enumerate(fc):
        Z = f["Z"]
        if Z.shape[0] != N:          # keep scenario count fixed (AR lag drops 1 row)
            Z = Z[-N:]
        Rsc = f["mu"] + f["sigma"] * Z
        if f["refit"] or k == 0:     # radius and kappas updated with the parameters
            e0 = eps_base(Z, cfg.eps_bootstrap_B, cfg.eps_quantile, cfg.eps_mean_block, rng)
            ke, kc = kappas_elliptical(np.median(f["nu"]), d, cfg.alpha, K=cfg.poss_mc_draws)
        eps0[k], kE[k], kC[k] = e0, ke, kc
        S = np.cov(Z, rowvar=False)
        A, _, _ = possibility_cuts(Z, cfg.poss_gammas)

        jobs = [("Nominal", "nw", 0.0)] + [(f"Wass x{s:g}", "nw", s * e0) for s in scales]
        for name, _, eps in jobs:
            nw.set_data(Rsc, f["sigma"], eps)
            for j, lam in enumerate(lams):
                w, st = nw.solve(lam)
                inacc[name] += st == "optimal_inaccurate"
                if w is None or st not in ("optimal", "optimal_inaccurate"):
                    bad[name] += 1; w = W[name][k - 1, j] if k else np.ones(d) / d
                W[name][k, j] = clean(w)
        pa.set_data(f["mu"], f["sigma"], S, ke, kc)
        pb.set_data(Rsc, A)
        for name, mdl in (("Poss-A", pa), ("Poss-B", pb)):
            for j, lam in enumerate(lams):
                w, st = mdl.solve(lam)
                inacc[name] += st == "optimal_inaccurate"
                if w is None or st not in ("optimal", "optimal_inaccurate"):
                    bad[name] += 1; w = W[name][k - 1, j] if k else np.ones(d) / d
                W[name][k, j] = clean(w)
        if verbose and k % 50 == 0:
            el = time.time() - t0
            print(f"  decision {k+1}/{T}  {f['date'].date()}  eps0={e0:.3f} kE={ke:.2f} kC={kc:.2f}"
                  f"  {el:.0f}s elapsed, ~{el/(k+1)*(T-k-1)/60:.1f} min left", flush=True)
    info = {"names": names, "lambdas": lams, "dates": [f["date"] for f in fc],
            "eval_dates": [R.index[f["i"] + 1] for f in fc], "eps0": eps0, "kE": kE, "kC": kC,
            "bad_solves": bad, "inaccurate_solves": inacc,
            "solves_per_model": T * len(lams), "N": N, "d": d}
    if cache is not None:
        np.savez_compressed(cache, W=np.array(W, dtype=object), info=np.array(info, dtype=object))
    return W, info


def realized_returns(R: pd.DataFrame, fc: list, W: dict, cfg: Config):
    """Simple portfolio returns (gross and net of costs) for every model and lambda.

    The weights chosen at the end of week i are held over week i+1.
    Turnover is measured against the drifted previous weights.
    """
    S = np.expm1(R.values)                                  # simple returns
    nxt = np.array([f["i"] + 1 for f in fc])
    Snext = S[nxt]                                          # (T, d)
    out = {}
    c = cfg.cost_bp / 1e4
    for m, Wm in W.items():
        T, nl, d = Wm.shape
        gross = np.einsum("tld,td->tl", Wm, Snext)
        to = np.zeros((T, nl))
        for t in range(1, T):
            prev = Wm[t - 1] * (1 + Snext[t - 1])
            prev /= prev.sum(1, keepdims=True)
            to[t] = np.abs(Wm[t] - prev).sum(1)
        to[0] = np.abs(Wm[0] - 1.0 / d).sum(1)
        out[m] = {"gross": gross, "net": gross - c * to, "turnover": to}
    # 1/N benchmark (rebalanced weekly, same cost model)
    T, d = Snext.shape
    w = np.full(d, 1.0 / d)
    g = Snext @ w
    to = np.zeros(T)
    for t in range(1, T):
        prev = w * (1 + Snext[t - 1]); prev /= prev.sum()
        to[t] = np.abs(w - prev).sum()
    out["1/N"] = {"gross": np.repeat(g[:, None], len(cfg.lambdas), 1),
                  "net": np.repeat((g - c * to)[:, None], len(cfg.lambdas), 1),
                  "turnover": np.repeat(to[:, None], len(cfg.lambdas), 1)}
    return out
