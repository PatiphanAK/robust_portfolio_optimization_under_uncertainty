"""Backtest: return timing, turnover/cost arithmetic on a hand-worked example,
and an end-to-end mini backtest (solve health, look-ahead, feasibility)."""
from __future__ import annotations
try:
    from .common import Checker, fixture
except ImportError:
    from common import Checker, fixture
from dataclasses import replace
import numpy as np
import pandas as pd
from forecast import run_forecasts
from backtest import run_backtest, realized_returns


def run(ck: Checker, fx=None):
    fx = fx or fixture()
    cfg = fx.cfg

    ck.section("8. return timing: weights at week i earn returns of week i+1")
    rng = np.random.default_rng(4)
    T0, d = 12, 3
    S = rng.normal(0, 0.03, (T0, d))
    R = pd.DataFrame(np.log1p(S), index=pd.date_range("2020-01-03", periods=T0, freq="W-FRI"))
    idx = [2, 3, 5, 8]                                   # decision weeks (not contiguous)
    fc = [{"i": i} for i in idx]
    Wm = rng.dirichlet(np.ones(d), (len(idx), 2))         # (T, n_lambda, d)
    out = realized_returns(R, fc, {"M": Wm}, replace(cfg, lambdas=(0.0, 1.0)))
    exp = np.einsum("tld,td->tl", Wm, S[np.array(idx) + 1])
    ck("gross[t] == w_t . r_{i_t + 1}", np.allclose(out["M"]["gross"], exp))
    ck("gross does not use same-week returns",
       not np.allclose(out["M"]["gross"], np.einsum("tld,td->tl", Wm, S[idx])))

    ck.section("9. turnover and cost on a hand-worked 2-asset example")
    S = np.array([[0.0, 0.0], [0.10, -0.10], [0.0, 0.20], [0.05, 0.05]])
    R = pd.DataFrame(np.log1p(S), index=pd.date_range("2020-01-03", periods=4, freq="W-FRI"))
    fc = [{"i": 0}, {"i": 1}, {"i": 2}]
    Wm = np.array([[[0.5, 0.5]], [[0.8, 0.2]], [[0.8, 0.2]]])
    c1 = replace(cfg, lambdas=(0.5,), cost_bp=10.0)
    o = realized_returns(R, fc, {"M": Wm}, c1)
    # by hand: gross = [0, .04, .05]
    # to_1: drifted [.55,.45] -> |.8-.55|+|.2-.45| = .5
    # to_2: drifted [.8,.24]/1.04 -> 2*(.8-.8/1.04) = .064/1.04
    g_h = np.array([0.0, 0.04, 0.05]); to_h = np.array([0.0, 0.5, 0.064 / 1.04])
    ck("gross by hand", np.allclose(o["M"]["gross"][:, 0], g_h))
    ck("turnover by hand (drifted weights)", np.allclose(o["M"]["turnover"][:, 0], to_h),
       f"{o['M']['turnover'][:, 0].round(6)}")
    ck("net = gross - 10bp * turnover", np.allclose(o["M"]["net"][:, 0], g_h - 1e-3 * to_h))
    # 1/N: drifted [.55,.45] -> .1 ; [.5,.6]/1.1 -> 2*(.6/1.1-.5) = .1/1.1
    ck("1/N turnover by hand", np.allclose(o["1/N"]["turnover"][:, 0], [0, 0.1, 0.1 / 1.1]))

    ck.section("10. end-to-end mini backtest: solve health, feasibility, look-ahead")
    cs = replace(cfg, eps_bootstrap_B=5, poss_mc_draws=20_000)
    W, info = run_backtest(fx.R, fx.fc, cs, appendix=False, verbose=False)
    ck("no failed solves", sum(info["bad_solves"].values()) == 0, f"{info['bad_solves']}")
    ck("inaccurate solves tracked", "inaccurate_solves" in info, f"{info.get('inaccurate_solves')}")
    allW = np.concatenate([w.reshape(-1, fx.d) for w in W.values()])
    ck("all weights finite, on simplex",
       np.isfinite(allW).all() and allW.min() >= 0 and np.allclose(allW.sum(1), 1))
    ck("evaluation date strictly after decision date",
       all(e > dd for e, dd in zip(info["eval_dates"], info["dates"])))
    i0 = fx.fc[0]["i"]
    R2 = fx.R.copy(); R2.iloc[i0 + 1:] = R2.iloc[i0 + 1:] * 5 + 0.1
    fc2 = run_forecasts(R2, cfg, verbose=False)
    W2, _ = run_backtest(R2, fc2, cs, appendix=False, verbose=False)
    same0 = all(np.allclose(W[m][0], W2[m][0]) for m in W)
    diff1 = any(not np.allclose(W[m][1], W2[m][1]) for m in W)
    ck("decision 0 weights unchanged by future returns", same0)
    ck("look-ahead test has power (decision 1 does change)", diff1)


if __name__ == "__main__":
    import sys
    ck = Checker(); run(ck)
    print("\nALL PASS" if ck.ok else f"\n{ck.n_fail} CHECK(S) FAILED"); sys.exit(0 if ck.ok else 1)
