"""Live demo (~30 s): radius eps changes -> portfolio changes.

Needs the cache from run_exp1.py (forecasts). Shows
  1. GARCH one-step volatility forecasts for the chosen week
  2. Wasserstein weights as EPS_SCALES grows (stacked area)  -> moves toward 1/N
  3. Poss-A vs Poss-B weights across lambda (front collapse vs real trade-off)
Usage: python exp1_mini.py [--synthetic] [--date 2020-03-20]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG
from ambiguity import eps_base, kappas_elliptical, possibility_cuts
from optimize import NominalWass, PossA, PossB, clean

EPS_SCALES = np.round(np.linspace(0, 3, 13), 2)   # edit live
LAMBDA = 0.5

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    cfg = CFG
    tag = "synthetic" if a.synthetic else "real"
    fc = list(np.load(ROOT / "cache" / tag / "forecasts.npz", allow_pickle=True)["out"])
    dates = pd.DatetimeIndex([f["date"] for f in fc])
    k = len(fc) - 1 if a.date is None else int(np.argmin(np.abs(dates - pd.Timestamp(a.date))))
    f = fc[k]
    Z, mu, sig = f["Z"], f["mu"], f["sigma"]
    N, d = Z.shape
    tick = list(cfg.tickers)
    print(f"decision week: {f['date'].date()}")
    print("GARCH 1-week vol forecast (% ann.):",
          {t: round(float(100 * s * np.sqrt(52)), 1) for t, s in zip(tick, sig)})

    Rsc = mu + sig * Z
    e0 = eps_base(Z, 50, cfg.eps_quantile, cfg.eps_mean_block, np.random.default_rng(cfg.seed))
    print(f"base radius eps0 = {e0:.3f}")
    nw = NominalWass(N, d, cfg.alpha, cfg.ridge)
    Wg = []
    for s in EPS_SCALES:
        nw.set_data(Rsc, sig, s * e0)
        Wg.append(clean(nw.solve(LAMBDA)[0]))
    Wg = np.array(Wg)

    S = np.cov(Z, rowvar=False)
    kE, kC = kappas_elliptical(np.median(f["nu"]), d, cfg.alpha)
    pa = PossA(d, cfg.ridge); pa.set_data(mu, sig, S, kE, kC)
    A, b, _ = possibility_cuts(Z, cfg.poss_gammas)
    pb = PossB(N, d, len(b), b, cfg.alpha, cfg.ridge); pb.set_data(Rsc, A)
    WA = np.array([clean(pa.solve(l)[0]) for l in cfg.lambdas])
    WB = np.array([clean(pb.solve(l)[0]) for l in cfg.lambdas])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    ax[0].stackplot(EPS_SCALES, Wg.T, labels=tick, step="post")
    ax[0].axhline(1, color="k", lw=.5)
    ax[0].set_xlabel("eps / eps0"); ax[0].set_ylabel("weight")
    ax[0].set_title(f"Wasserstein, lambda={LAMBDA}: larger eps -> toward 1/N", fontsize=9)
    for axi, Wm, ttl in ((ax[1], WA, "Poss-A (elliptical): weights barely change"),
                         (ax[2], WB, "Poss-B (scenario): real trade-off")):
        axi.stackplot(cfg.lambdas, Wm.T, step="post")
        axi.set_xlabel("lambda (0 = CVaR only, 1 = mean only)"); axi.set_title(ttl, fontsize=9)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=9, fontsize=8, frameon=False)
    fig.suptitle(f"Decision week {f['date'].date()}", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / "results" / tag / "fig_demo_weights.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("max |w - 1/N| at largest eps:", round(float(np.abs(Wg[-1] - 1 / d).max()), 3))
    print("saved", out)


if __name__ == "__main__":
    main()
