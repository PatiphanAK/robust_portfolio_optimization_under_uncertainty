"""Ambiguity-set ingredients, all computed from the training window only.

Wasserstein : base radius from a stationary bootstrap of residual vectors and
              the exact W1 distance (l2 ground metric, assignment problem).
Possibility A (elliptical): kappa_E = E[sqrt(M)], kappa_C = CVaR_alpha(sqrt(M))
              for the squared Mahalanobis radius M of a standardized
              multivariate t; gives closed forms (Lemma 2.1).
Possibility B (scenario): nested empirical alpha-cuts of the FHS scenarios,
              ranked by Mahalanobis distance; a distribution p on the scenarios is
              admissible iff p(C_k) >= 1 - gamma_k for every cut.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------- Wasserstein
def stationary_bootstrap_idx(n: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (circular)."""
    q = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < q else (idx[t - 1] + 1) % n
    return idx


def w1_exact(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact W1 between two equal-size uniform empirical measures (l2 cost)."""
    C = cdist(X, Y)
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


def eps_base(Z: np.ndarray, B: int, q: float, mean_block: float, rng) -> float:
    n = Z.shape[0]
    d = [w1_exact(Z, Z[stationary_bootstrap_idx(n, mean_block, rng)]) for _ in range(B)]
    return float(np.quantile(d, q))


# ---------------------------------------------------------------- Possibility A
def kappas_elliptical(nu: float, d: int, alpha: float, K: int = 400_000):
    """kappa_E = E[sqrt(M)], kappa_C = CVaR_alpha(sqrt(M)) (upper tail).

    z = sqrt((nu-2)/nu) * T, T multivariate t_nu with unit-diagonal correlation R
    => M = z' R^{-1} z = (nu-2)/nu * d * F(d, nu).
    Deterministic quantile grid instead of random draws.
    """
    nu = max(float(nu), 2.05)
    u = (np.arange(K) + 0.5) / K
    rad = np.sqrt((nu - 2) / nu * d * stats.f.ppf(u, d, nu))
    kE = rad.mean()
    kC = rad[u >= 1 - alpha].mean()
    return float(kE), float(kC)


# ---------------------------------------------------------------- Possibility B
def possibility_cuts(Z: np.ndarray, gammas) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, b, pi): A[k,s]=1 if scenario s lies in cut k, b[k]=1-gamma_k,
    pi[s] = empirical possibility degree of scenario s
    (probability-possibility transformation: share of scenarios at least as far)."""
    N = Z.shape[0]
    S = np.cov(Z, rowvar=False)
    Sinv = np.linalg.inv(S + 1e-10 * np.eye(S.shape[0]))
    dist = np.sqrt(np.einsum("ij,jk,ik->i", Z, Sinv, Z))      # centre = forecast (z=0)
    order = np.argsort(dist, kind="stable")
    rank = np.empty(N, dtype=int); rank[order] = np.arange(N)  # 0 = most central
    pi = (N - rank) / N
    A, b = [], []
    for g in gammas:
        m = int(np.ceil((1 - g) * N))
        A.append((rank < m).astype(float))
        b.append(1 - g)
    return np.array(A), np.array(b), pi
