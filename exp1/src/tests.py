"""Verification checks (Experiment 0 style). Run: python tests.py"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG
import data as D
from forecast import run_forecasts
from ambiguity import kappas_elliptical, possibility_cuts, eps_base
from optimize import NominalWass, PossA, PossB, objectives_fixed_w, emp_cvar, clean, is_conv_func
import metrics as M

rng = np.random.default_rng(0)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# --- toy scenario set from the real pipeline on synthetic data
cfg = CFG
R = D.synthetic_returns(cfg, n_weeks=cfg.train_window + 6, seed=3)
fc = run_forecasts(R, cfg, verbose=False)
f = fc[0]
Z, mu, sig = f["Z"], f["mu"], f["sigma"]
N, d = Z.shape
Rsc = mu + sig * Z
alpha = cfg.alpha
S = np.cov(Z, rowvar=False)

# 1. look-ahead: changing returns after the decision week must not change the forecast
R2 = R.copy(); R2.iloc[f["i"] + 1:] = R2.iloc[f["i"] + 1:] * 5 + 0.1
f2 = run_forecasts(R2, cfg, verbose=False)[0]
check("no look-ahead in forecasts", np.allclose(f["mu"], f2["mu"]) and np.allclose(f["sigma"], f2["sigma"]))

# 2. eps = 0 equals nominal; worst case non-decreasing in eps (fixed w)
w = rng.dirichlet(np.ones(d))
L = -Rsc @ w
F0 = objectives_fixed_w("wass", w, Rsc=Rsc, sigma=sig, alpha=alpha, eps=0.0)
check("Wasserstein eps=0 == nominal", np.allclose(F0, (L.mean(), emp_cvar(L, alpha))))
vals = [objectives_fixed_w("wass", w, Rsc=Rsc, sigma=sig, alpha=alpha, eps=e) for e in (0, .05, .1, .5)]
check("Wasserstein monotone in eps", bool(np.all(np.diff(np.array(vals), axis=0) >= -1e-12)))

# 3. solver objective equals closed-form objective at the optimum (Nominal / Wass)
nw = NominalWass(N, d, alpha, 0.0)
for eps in (0.0, 0.3):
    nw.set_data(Rsc, sig, eps)
    for lam in (0.0, 0.5, 1.0):
        ws, st = nw.solve(lam)
        F = objectives_fixed_w("wass", clean(ws), Rsc=Rsc, sigma=sig, alpha=alpha, eps=eps)
        check(f"Wass eps={eps} lam={lam} solver == formula", abs(nw.prob.value - (lam*F[0]+(1-lam)*F[1])) < 1e-6,
              f"({nw.prob.value:.6g} vs {lam*F[0]+(1-lam)*F[1]:.6g}, {st})")

# 4. Possibility B: dual LP value == primal worst case (max over admissible p)
A, b, pi = possibility_cuts(Z, cfg.poss_gammas)
check("Poss-B empirical p is admissible", np.all(A @ np.full(N, 1 / N) >= b - 1e-12))
pb = PossB(N, d, len(b), b, alpha, 0.0)
pb.set_data(Rsc, A)
for lam in (0.0, 0.5, 1.0):
    ws, st = pb.solve(lam)
    ws = clean(ws)
    F = objectives_fixed_w("possB", ws, Rsc=Rsc, alpha=alpha, A=A, b=b)
    check(f"Poss-B lam={lam} dual == primal", abs(pb.prob.value - (lam*F[0]+(1-lam)*F[1])) < 1e-5,
          f"({pb.prob.value:.6g} vs {lam*F[0]+(1-lam)*F[1]:.6g}, {st})")
Fb = objectives_fixed_w("possB", w, Rsc=Rsc, alpha=alpha, A=A, b=b)
Fn = (L.mean(), emp_cvar(L, alpha))
check("Poss-B worst case >= nominal", Fb[0] >= Fn[0] - 1e-9 and Fb[1] >= Fn[1] - 1e-9,
      f"E {Fn[0]:.5f}->{Fb[0]:.5f}, CVaR {Fn[1]:.5f}->{Fb[1]:.5f}")

# 5. Possibility A: kappas match Monte Carlo of a standardized multivariate t
nu = 6.0
kE, kC = kappas_elliptical(nu, d, alpha)
g = rng.chisquare(nu, 400_000) / nu
rad = np.sqrt(np.sum(rng.normal(size=(400_000, d))**2, 1) / g * (nu - 2) / nu)
q = np.quantile(rad, 1 - alpha)
check("Poss-A kappa_E vs MC", abs(kE - rad.mean()) < 0.01, f"{kE:.4f} vs {rad.mean():.4f}")
check("Poss-A kappa_C vs MC", abs(kC - rad[rad >= q].mean()) < 0.03, f"{kC:.4f} vs {rad[rad>=q].mean():.4f}")
pa = PossA(d, 0.0); pa.set_data(mu, sig, S, kE, kC)
ws, st = pa.solve(0.5)
FA = objectives_fixed_w("possA", clean(ws), Rsc=Rsc, sigma=sig, mu=mu, S=S, kE=kE, kC=kC)
check("Poss-A solver == formula", abs(pa.prob.value - 0.5 * sum(FA)) < 1e-7)

# 6. feasibility of weights
check("weights on simplex", abs(clean(ws).sum() - 1) < 1e-9 and clean(ws).min() >= 0)

# 7. eps base positive and finite
e0 = eps_base(Z, 20, 0.95, 4.0, rng)
check("eps base finite > 0", np.isfinite(e0) and e0 > 0, f"{e0:.3f}")

# 8. HV on a known front; Sharpe test sanity
check("HV 2-D exact", abs(M.hv2d(np.array([[0, 1], [1, 0], [.5, .5]]), np.array([2, 2])) - 3.25) < 1e-12)
x = rng.standard_normal(500) * 0.02 + 0.002
check("Sharpe test identical series", M.sharpe_diff_test(x, x + 1e-12 * rng.standard_normal(500))[1] > 0.9)
rej = 0
for _ in range(300):
    e = rng.multivariate_normal([0.002, 0.002], [[4e-4, 3e-4], [3e-4, 4e-4]], 400)
    rej += M.sharpe_diff_test(e[:, 0], e[:, 1])[1] < 0.05
check("Sharpe test size ~5% under H0", 0.02 <= rej / 300 <= 0.09, f"{rej/300:.3f}")

# 9. convexity of objectives via numerical Hessian (PSD at random simplex points)
P = rng.dirichlet(np.ones(d), 5)
check("is_conv_func sanity: ||w||^2 convex, -||w||^2 not",
      is_conv_func(lambda v: v @ v, P) and not is_conv_func(lambda v: -(v @ v), P))
for k, name in ((0, "F1"), (1, "F2")):
    fw = lambda v, k=k: objectives_fixed_w("wass", v, Rsc=Rsc, sigma=sig, alpha=alpha, eps=0.3)[k]
    ok, info = is_conv_func(fw, P, return_info=True)
    check(f"Wass {name} convex (Hessian PSD)", ok, f"min eig {info['lambda_min'].min():.2e}")
    fa = lambda v, k=k: objectives_fixed_w("possA", v, Rsc=Rsc, sigma=sig, mu=mu, S=S, kE=kE, kC=kC)[k]
    ok, info = is_conv_func(fa, P, return_info=True)
    check(f"Poss-A {name} convex (Hessian PSD)", ok, f"min eig {info['lambda_min'].min():.2e}")

print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
