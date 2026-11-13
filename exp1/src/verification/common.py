"""Shared fixture and PASS/FAIL reporter for the verification suite."""
from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import cvxpy as cp

from config import CFG
import data as D
from forecast import run_forecasts
from ambiguity import kappas_elliptical, possibility_cuts
from optimize import SOLVER


class Checker:
    def __init__(self):
        self.n_pass = self.n_fail = 0

    def __call__(self, name, cond, detail=""):
        cond = bool(cond)
        self.n_pass += cond
        self.n_fail += not cond
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
        return cond

    def section(self, title):
        print(f"\n== {title}")

    @property
    def ok(self):
        return self.n_fail == 0


@lru_cache(maxsize=None)
def fixture(seed: int = 3):
    """Synthetic data -> real forecast pipeline -> one decision's scenario set."""
    cfg = CFG
    R = D.synthetic_returns(cfg, n_weeks=cfg.train_window + 6, seed=seed)
    fc = run_forecasts(R, cfg, verbose=False)
    f = fc[0]
    Z, mu, sig = f["Z"], f["mu"], f["sigma"]
    N, d = Z.shape
    A, b, pi = possibility_cuts(Z, cfg.poss_gammas)
    kE, kC = kappas_elliptical(6.0, d, cfg.alpha)
    return SimpleNamespace(cfg=cfg, R=R, fc=fc, f=f, Z=Z, mu=mu, sig=sig, N=N, d=d,
                           Rsc=mu + sig * Z, alpha=cfg.alpha, S=np.cov(Z, rowvar=False),
                           A=A, b=b, pi=pi, kE=kE, kC=kC)


def possB_F(Rsc, w, A, b, alpha):
    """Poss-B (F1, F2) at a fixed w via the dual LPs (one LP each, fast)."""
    L = -Rsc @ w
    K = len(b)
    t = cp.Variable(); y = cp.Variable(K, nonneg=True)
    f1 = cp.Problem(cp.Minimize(t - b @ y), [t - A.T @ y >= L]).solve(solver=SOLVER)
    t = cp.Variable(); y = cp.Variable(K, nonneg=True); eta = cp.Variable()
    f2 = cp.Problem(cp.Minimize(eta + (t - b @ y) / alpha),
                    [t - A.T @ y >= L - eta, t - A.T @ y >= 0]).solve(solver=SOLVER)
    return f1, f2


def F_pair(model, w, fx, eps=0.3):
    """(F1, F2) of `model` in {'nominal','wass','possA','possB'} at fixed w."""
    from optimize import objectives_fixed_w
    if model == "possB":
        return possB_F(fx.Rsc, w, fx.A, fx.b, fx.alpha)
    if model in ("nominal", "wass"):
        return objectives_fixed_w("wass", w, Rsc=fx.Rsc, sigma=fx.sig, alpha=fx.alpha,
                                  eps=0.0 if model == "nominal" else eps)
    return objectives_fixed_w("possA", w, Rsc=fx.Rsc, sigma=fx.sig, mu=fx.mu, S=fx.S,
                              kE=fx.kE, kC=fx.kC)


def solver_for(model, fx, ridge=0.0, eps=0.3):
    """Return a callable lam -> (w, status, objective value) for `model`."""
    from optimize import NominalWass, PossA, PossB
    if model in ("nominal", "wass"):
        m = NominalWass(fx.N, fx.d, fx.alpha, ridge)
        m.set_data(fx.Rsc, fx.sig, 0.0 if model == "nominal" else eps)
    elif model == "possA":
        m = PossA(fx.d, ridge); m.set_data(fx.mu, fx.sig, fx.S, fx.kE, fx.kC)
    else:
        m = PossB(fx.N, fx.d, len(fx.b), fx.b, fx.alpha, ridge); m.set_data(fx.Rsc, fx.A)

    def solve(lam):
        w, st = m.solve(lam)
        return w, st, m.prob.value
    return solve


MODELS = ("nominal", "wass", "possA", "possB")
