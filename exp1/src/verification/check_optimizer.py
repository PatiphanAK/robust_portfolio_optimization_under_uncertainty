"""Optimizer correctness: global optimality, Pareto-front monotonicity,
Poss-A formula at every lambda, effect of the production ridge."""
from __future__ import annotations
try:
    from .common import Checker, fixture, F_pair, solver_for, MODELS
except ImportError:
    from common import Checker, fixture, F_pair, solver_for, MODELS
import numpy as np
from optimize import clean


def candidates(d, rng, n):
    """Random simplex points (dense + sparse) plus all vertices."""
    return np.vstack([np.eye(d), rng.dirichlet(np.ones(d), n // 2),
                      rng.dirichlet(np.full(d, 0.2), n - n // 2)])


def run(ck: Checker, fx=None):
    fx = fx or fixture()
    rng = np.random.default_rng(1)

    ck.section("1. global optimality (solution beats random feasible portfolios)")
    for model in MODELS:
        solve = solver_for(model, fx)
        C = candidates(fx.d, rng, 60 if model == "possB" else 2000)
        FC = np.array([F_pair(model, c, fx) for c in C])
        for lam in (0.0, 0.5, 1.0):
            w, st, _ = solve(lam)
            g_opt = np.dot((lam, 1 - lam), F_pair(model, clean(w), fx))
            g_best = (FC @ np.array([lam, 1 - lam])).min()
            ck(f"{model} lam={lam} optimal", g_opt <= g_best + 1e-6 * max(1, abs(g_best)),
               f"(g*={g_opt:.6g}, best random={g_best:.6g}, {st})")

    ck.section("2. Pareto front monotone in lambda (F1 non-increasing, F2 non-decreasing)")
    lams = np.array(fx.cfg.lambdas)
    for model in MODELS:
        solve = solver_for(model, fx)
        F = np.array([F_pair(model, clean(solve(l)[0]), fx) for l in lams])
        tol = 1e-6 * max(1.0, np.abs(F).max())
        d1, d2 = np.diff(F[:, 0]), np.diff(F[:, 1])
        ck(f"{model} front monotone", np.all(d1 <= tol) and np.all(d2 >= -tol),
           f"(max dF1={d1.max():.2e}, min dF2={d2.min():.2e})")

    ck.section("3. Poss-A solver == closed form at every lambda")
    solve = solver_for("possA", fx)
    errs = []
    for lam in lams:
        w, st, val = solve(lam)
        errs.append(abs(val - np.dot((lam, 1 - lam), F_pair("possA", clean(w), fx))))
    ck("Poss-A solver == formula, all lambdas", max(errs) < 1e-7, f"(max err {max(errs):.2e})")

    ck.section("4. production ridge changes the objective only negligibly")
    r = fx.cfg.ridge
    for model in MODELS:
        s0, sr = solver_for(model, fx, ridge=0.0), solver_for(model, fx, ridge=r)
        gaps = []
        for lam in (0.0, 0.5, 1.0):
            c = np.array([lam, 1 - lam])
            g0 = c @ F_pair(model, clean(s0(lam)[0]), fx)
            gr = c @ F_pair(model, clean(sr(lam)[0]), fx)
            gaps.append(gr - g0)
        # 0 <= g(w_ridge) - g(w_0) <= ridge * ||w_0||^2 <= ridge
        ck(f"{model} ridge={r:g} gap", min(gaps) >= -1e-6 and max(gaps) <= r + 1e-6,
           f"(gaps {min(gaps):.1e}..{max(gaps):.1e})")


if __name__ == "__main__":
    import sys
    ck = Checker(); run(ck)
    print("\nALL PASS" if ck.ok else f"\n{ck.n_fail} CHECK(S) FAILED"); sys.exit(0 if ck.ok else 1)
