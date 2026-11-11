"""Robust bi-objective portfolio problems, solved by weighted-sum scalarization.

Scenarios (FHS):  xi_s = mu + sigma * z_s,   loss L_s(w) = -xi_s' w,  w in simplex.
For every model k:
    F1 = worst-case expected loss,  F2 = worst-case CVaR_alpha of loss,
    w*_lambda = argmin  lambda F1 + (1-lambda) F2 + ridge ||w||^2.

Nominal       : empirical E and CVaR (Rockafellar-Uryasev LP).
Wasserstein   : W1 ball of radius eps around the empirical residual distribution,
                l2 ground metric in residual space, unbounded support:
                F1 = E_hat + eps ||sigma*w||_2,  F2 = CVaR_hat + (eps/alpha) ||sigma*w||_2.
Possibility A : elliptical forecast distribution, Sigma = D S D:
                F1 = -mu'w + kE ||S^{1/2} D w||,  F2 = -mu'w + kC ||S^{1/2} D w||.
Possibility B : p on scenarios with p(C_k) >= b_k (nested cuts); LP duals:
                F1 = min t - b'y   s.t. t - A'y >= L,           y >= 0
                F2 = min eta + (t - b'y)/alpha
                                 s.t. t - A'y >= L - eta, t - A'y >= 0, y >= 0.
All parameters enter in DPP form so each problem is compiled once.
"""
from __future__ import annotations
import numpy as np
import cvxpy as cp

SOLVER = "CLARABEL"


class NominalWass:
    """Nominal is the special case eps = 0."""

    def __init__(self, N, d, alpha, ridge):
        self.alpha, self.N = alpha, N
        self.w = cp.Variable(d, nonneg=True)
        eta = cp.Variable(); u = cp.Variable(N, nonneg=True)
        self.Rsc = cp.Parameter((N, d))
        self.rbar_l = cp.Parameter(d)          # lambda * mean scenario
        self.s_c = cp.Parameter(d, nonneg=True)  # (lambda*eps + (1-lambda)*eps/alpha) * sigma
        self.cl = cp.Parameter(nonneg=True)      # 1 - lambda
        self.cu = cp.Parameter(nonneg=True)      # (1 - lambda)/(alpha N)
        obj = (-self.rbar_l @ self.w + cp.norm(cp.multiply(self.s_c, self.w), 2)
               + self.cl * eta + self.cu * cp.sum(u) + ridge * cp.sum_squares(self.w))
        cons = [cp.sum(self.w) == 1, u >= -self.Rsc @ self.w - eta]
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def set_data(self, Rsc, sigma, eps):
        self.Rsc.value = Rsc
        self._rbar, self._sig, self._eps = Rsc.mean(0), sigma, eps

    def solve(self, lam):
        a = self.alpha
        self.rbar_l.value = lam * self._rbar
        self.s_c.value = (lam * self._eps + (1 - lam) * self._eps / a) * self._sig
        self.cl.value = 1 - lam
        self.cu.value = (1 - lam) / (a * self.N)
        self.prob.solve(solver=SOLVER)
        return self.w.value, self.prob.status


class PossA:
    def __init__(self, d, ridge):
        self.w = cp.Variable(d, nonneg=True)
        self.mu = cp.Parameter(d)
        self.G = cp.Parameter((d, d))            # coef * S^{1/2} D
        obj = -self.mu @ self.w + cp.norm(self.G @ self.w, 2) + ridge * cp.sum_squares(self.w)
        self.prob = cp.Problem(cp.Minimize(obj), [cp.sum(self.w) == 1])

    def set_data(self, mu, sigma, S, kE, kC):
        self.mu.value = mu
        self._G = np.linalg.cholesky(S + 1e-12 * np.eye(len(mu))).T @ np.diag(sigma)
        self._kE, self._kC = kE, kC

    def solve(self, lam):
        self.G.value = (lam * self._kE + (1 - lam) * self._kC) * self._G
        self.prob.solve(solver=SOLVER)
        return self.w.value, self.prob.status


class PossB:
    def __init__(self, N, d, K, b, alpha, ridge):
        self.w = cp.Variable(d, nonneg=True)
        t1 = cp.Variable(); y1 = cp.Variable(K, nonneg=True)
        t2 = cp.Variable(); y2 = cp.Variable(K, nonneg=True); eta = cp.Variable()
        self.Rsc = cp.Parameter((N, d))
        self.At = cp.Parameter((N, K), nonneg=True)    # A transpose
        self.lam = cp.Parameter(nonneg=True)
        self.cl = cp.Parameter(nonneg=True)
        L = -self.Rsc @ self.w
        F1 = t1 - b @ y1
        F2 = eta + (t2 - b @ y2) / alpha
        obj = self.lam * t1 - self.lam * (b @ y1) + self.cl * eta \
            + (self.cl / alpha) * t2 - (self.cl / alpha) * (b @ y2) + ridge * cp.sum_squares(self.w)
        cons = [cp.sum(self.w) == 1,
                t1 - self.At @ y1 >= L,
                t2 - self.At @ y2 >= L - eta,
                t2 - self.At @ y2 >= 0]
        self.prob = cp.Problem(cp.Minimize(obj), cons)
        self._F1, self._F2 = F1, F2

    def set_data(self, Rsc, A):
        self.Rsc.value = Rsc
        self.At.value = A.T

    def solve(self, lam):
        self.lam.value = lam
        self.cl.value = 1 - lam
        self.prob.solve(solver=SOLVER)
        return self.w.value, self.prob.status


def clean(w):
    w = np.clip(np.asarray(w, float), 0, None)
    return w / w.sum()


# ------------------------------------------------ objective evaluation for checks
def emp_cvar(L, alpha):
    q = np.quantile(L, 1 - alpha, method="inverted_cdf")
    return q + np.mean(np.maximum(L - q, 0)) / alpha


def objectives_fixed_w(model, w, *, Rsc, sigma=None, alpha=0.05, eps=0.0, mu=None, S=None,
                       kE=None, kC=None, A=None, b=None):
    """Closed-form / LP evaluation of (F1, F2) for a fixed portfolio (used in tests)."""
    L = -Rsc @ w
    if model == "wass":
        pen = eps * np.linalg.norm(sigma * w)
        return L.mean() + pen, emp_cvar(L, alpha) + pen / alpha
    if model == "possA":
        s = np.linalg.norm(np.linalg.cholesky(S).T @ (sigma * w))
        return -mu @ w + kE * s, -mu @ w + kC * s
    if model == "possB":   # primal: maximise over admissible p (direct, no duality)
        p = cp.Variable(len(L), nonneg=True)
        cons = [cp.sum(p) == 1, A @ p >= b]
        f1 = cp.Problem(cp.Maximize(L @ p), cons).solve(solver=SOLVER)
        # min_eta max_p [eta + E_p(L-eta)+/alpha]; the outer function is convex
        # piecewise-linear in eta with kinks at scenario losses, so scan those.
        vals = []
        for eta in np.unique(L):
            v = cp.Problem(cp.Maximize(np.maximum(L - eta, 0) @ p), cons).solve(solver=SOLVER)
            vals.append(eta + v / alpha)
        # minimax equality: min_eta max_p = max_p min_eta (convex-concave, compact p set)
        return f1, min(vals)
    raise ValueError(model)
