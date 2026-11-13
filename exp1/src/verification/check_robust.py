"""Robust formulas: Wasserstein closed form is attained (tight) and is an upper
bound; possibility cuts are nested; elliptical kappas are sane."""
from __future__ import annotations
try:
    from .common import Checker, fixture
except ImportError:
    from common import Checker, fixture
import numpy as np
from scipy import stats
from scipy.special import gammaln
from ambiguity import w1_exact, kappas_elliptical, possibility_cuts
from optimize import objectives_fixed_w, emp_cvar


def run(ck: Checker, fx=None):
    fx = fx or fixture()
    rng = np.random.default_rng(2)
    a, eps = fx.alpha, 0.3
    # use N' scenarios with alpha*N' integer so empirical CVaR = mean of top k losses
    k = int(np.floor(a * fx.N)); Np = int(round(k / a))
    Z, mu, sig = fx.Z[:Np], fx.mu, fx.sig
    Rsc = mu + sig * Z

    ck.section("5. Wasserstein closed form is attained and is an upper bound")
    for trial in range(3):
        w = rng.dirichlet(np.ones(fx.d))
        F1, F2 = objectives_fixed_w("wass", w, Rsc=Rsc, sigma=sig, alpha=a, eps=eps)
        u = sig * w / np.linalg.norm(sig * w)          # worst direction in residual space
        loss = lambda Zq: -(mu + sig * Zq) @ w

        Z1 = Z - eps * u                               # translate all: W1 = eps
        ck(f"[w{trial}] F1 attained by translation", abs(loss(Z1).mean() - F1) < 1e-12,
           f"({loss(Z1).mean():.6g} vs {F1:.6g})")
        ck(f"[w{trial}] translation has W1 = eps", abs(w1_exact(Z, Z1) - eps) < 1e-9)

        top = np.argsort(loss(Z))[-k:]                 # worst alpha-tail
        Z2 = Z.copy(); Z2[top] -= (eps / a) * u        # cost k/N' * eps/alpha = eps
        c2 = emp_cvar(loss(Z2), a)
        ck(f"[w{trial}] F2 attained by tail shift", abs(c2 - F2) < 1e-10, f"({c2:.6g} vs {F2:.6g})")
        ck(f"[w{trial}] tail shift has W1 <= eps", w1_exact(Z, Z2) <= eps + 1e-9)

        worst1 = worst2 = -np.inf                      # random members of the ball
        for _ in range(200):
            r = rng.exponential(size=Np); r *= eps / r.mean()
            v = rng.standard_normal((Np, fx.d)); v /= np.linalg.norm(v, axis=1, keepdims=True)
            Lq = loss(Z + r[:, None] * v)
            worst1, worst2 = max(worst1, Lq.mean()), max(worst2, emp_cvar(Lq, a))
        ck(f"[w{trial}] random ball members <= closed form",
           worst1 <= F1 + 1e-12 and worst2 <= F2 + 1e-12,
           f"(E {worst1:.4g}<={F1:.4g}, CVaR {worst2:.4g}<={F2:.4g})")

    ck.section("6. possibility cuts nested and consistent")
    A, b, pi = fx.A, fx.b, fx.pi
    g = np.array(fx.cfg.poss_gammas)
    ck("b == 1 - gamma", np.allclose(b, 1 - g))
    ck("cut sizes == ceil((1-gamma) N)",
       np.array_equal(A.sum(1).astype(int), np.ceil((1 - g) * fx.N - 1e-9).astype(int)))
    order = np.argsort(A.sum(1))
    ck("cuts nested", all(np.all(A[order[i]] <= A[order[i + 1]]) for i in range(len(b) - 1)))
    Sinv = np.linalg.inv(fx.S + 1e-10 * np.eye(fx.d))
    dist = np.sqrt(np.einsum("ij,jk,ik->i", fx.Z, Sinv, fx.Z))
    o = np.argsort(dist)
    ck("possibility pi decreasing in distance, in (0,1]",
       np.all(np.diff(pi[o]) <= 0) and pi.max() == 1 and pi.min() > 0)
    ck("each cut = most central scenarios",
       all(dist[A[i] == 1].max() <= dist[A[i] == 0].min() for i in range(len(b)) if A[i].min() == 0))
    A2, b2, _ = possibility_cuts(fx.Z * 3.0 + 0.0, fx.cfg.poss_gammas)
    ck("cuts invariant to residual scale", np.array_equal(A, A2))

    ck.section("7. elliptical kappas")
    d = fx.d
    for nu in (3.0, 6.0, 30.0):
        kE, kC = kappas_elliptical(nu, d, a)
        ck(f"nu={nu:g} kappa_C >= kappa_E > 0", kC >= kE > 0, f"({kE:.4f}, {kC:.4f})")
    kE, kC = kappas_elliptical(1e7, d, a)
    u = (np.arange(400_000) + 0.5) / 400_000
    chi = stats.chi.ppf(u, d)
    kE_g = np.exp(0.5 * np.log(2) + gammaln((d + 1) / 2) - gammaln(d / 2))
    ck("nu->inf: kappa_E -> E[chi_d]", abs(kE - kE_g) < 1e-3, f"({kE:.5f} vs {kE_g:.5f})")
    ck("nu->inf: kappa_C -> CVaR(chi_d)", abs(kC - chi[u >= 1 - a].mean()) < 1e-3,
       f"({kC:.5f} vs {chi[u >= 1 - a].mean():.5f})")
    kEs = [kappas_elliptical(6.0, dd, a)[0] for dd in (1, 3, 9, 20)]
    ck("kappa_E increasing in dimension", np.all(np.diff(kEs) > 0))
    kCs = [kappas_elliptical(nu, d, a)[1] for nu in (3.0, 6.0, 30.0)]
    ck("kappa_C decreasing in nu (heavier tail -> larger)", np.all(np.diff(kCs) < 0))


if __name__ == "__main__":
    import sys
    ck = Checker(); run(ck)
    print("\nALL PASS" if ck.ok else f"\n{ck.n_fail} CHECK(S) FAILED"); sys.exit(0 if ck.ok else 1)
