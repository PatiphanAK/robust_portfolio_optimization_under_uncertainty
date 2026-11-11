"""Out-of-sample metrics, Pareto-front hypervolume and significance tests."""
from __future__ import annotations
import numpy as np
from scipy import stats


def sharpe(r, m=52):
    s = r.std(ddof=1)
    return np.nan if s == 0 else r.mean() / s * np.sqrt(m)


def max_drawdown(r):
    w = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return float(np.max(1 - w / peak))


def cvar_loss(r, alpha):
    """CVaR_alpha of the loss -r (positive = bad), Rockafellar-Uryasev on the sample."""
    L = -np.asarray(r)
    q = np.quantile(L, 1 - alpha, method="inverted_cdf")
    return float(q + np.mean(np.maximum(L - q, 0)) / alpha)


def summary(r, to, alpha, m=52):
    return {"Ann. return %": 100 * ((1 + r).prod() ** (m / len(r)) - 1),
            "Ann. vol %": 100 * r.std(ddof=1) * np.sqrt(m),
            "Sharpe": sharpe(r, m),
            "MDD %": 100 * max_drawdown(r),
            "CVaR %": 100 * cvar_loss(r, alpha),
            "Turnover": float(np.mean(to))}


# ------------------------------------------------------------------ hypervolume
def nondominated(P):
    P = np.asarray(P)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if np.any(np.all(P <= P[i], 1) & np.any(P < P[i], 1)):
            keep[i] = False
    return P[keep]


def hv2d(P, ref):
    """Exact 2-D hypervolume (minimisation) of a point set w.r.t. reference point."""
    P = nondominated(P)
    P = P[np.all(P < ref, 1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in P:
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return float(hv)


def oos_front(ret_tl, alpha):
    """Realised (mean loss, CVaR loss) of each lambda-portfolio: one point per lambda."""
    return np.array([[-ret_tl[:, j].mean(), cvar_loss(ret_tl[:, j], alpha)]
                     for j in range(ret_tl.shape[1])])


def normalised_hv(fronts: dict, ref):
    """Min-max normalise using all fronts jointly (common scale), then HV per model."""
    allp = np.vstack(list(fronts.values()))
    lo, hi = allp.min(0), allp.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return {m: hv2d((P - lo) / span, np.array(ref)) for m, P in fronts.items()}, (lo, span)


# ------------------------------------------------------------------ bootstrap
def stationary_idx(T, mean_block, rng):
    q = 1.0 / mean_block
    idx = np.empty(T, int); idx[0] = rng.integers(T)
    for t in range(1, T):
        idx[t] = rng.integers(T) if rng.random() < q else (idx[t - 1] + 1) % T
    return idx


def hv_diff_ci(rets: dict, base: str, alpha, ref, B, mean_block, rng, level=0.95):
    """Block-bootstrap CI of HV(model) - HV(base); the same resampled weeks for all models."""
    T = next(iter(rets.values())).shape[0]
    diffs = {m: [] for m in rets if m != base}
    for _ in range(B):
        idx = stationary_idx(T, mean_block, rng)
        fr = {m: oos_front(r[idx], alpha) for m, r in rets.items()}
        hv, _ = normalised_hv(fr, ref)
        for m in diffs:
            diffs[m].append(hv[m] - hv[base])
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {m: (float(np.quantile(v, lo)), float(np.quantile(v, hi))) for m, v in diffs.items()}


# ------------------------------------------------------------------ Sharpe test
def sharpe_diff_test(r1, r2):
    """Ledoit & Wolf (2008) HAC test of H0: SR1 = SR2 (delta method, Bartlett kernel,
    Newey-West automatic lag). Returns (diff in per-period SR, two-sided p-value)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    T = len(r1)
    m1, m2 = r1.mean(), r2.mean()
    g1, g2 = (r1**2).mean(), (r2**2).mean()
    s1, s2 = np.sqrt(g1 - m1**2), np.sqrt(g2 - m2**2)
    diff = m1 / s1 - m2 / s2
    Y = np.column_stack([r1 - m1, r2 - m2, r1**2 - g1, r2**2 - g2])
    L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    Psi = Y.T @ Y / T
    for l in range(1, L + 1):
        G = Y[l:].T @ Y[:-l] / T
        Psi += (1 - l / (L + 1)) * (G + G.T)
    grad = np.array([g1 / s1**3, -g2 / s2**3, -m1 / (2 * s1**3), m2 / (2 * s2**3)])
    v = grad @ Psi @ grad / T
    if not np.isfinite(v) or v <= 1e-18:          # (near-)identical strategies
        return float(diff), 1.0
    p = 2 * (1 - stats.norm.cdf(abs(diff) / np.sqrt(v)))
    return float(diff), float(p)


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(items); adj = {}; running = 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (n - k) * p))
        adj[name] = running
    return adj
