"""POST-HOC diagnostic of the lambda-fronts (not pre-registered).

Reads the cached in-sample weights from run_exp1.py (cache/<tag>/weights_<fingerprint>.npz);
no optimisation is re-run, and the config and results tables are not changed.

Per model, over all decision weeks:
  a) number of distinct portfolios among the 11 lambda values. Going through lambda in
     order, a portfolio counts as new if its L1 distance to every portfolio already
     counted is > 1e-3. Reports mean and median.
  b) mean L1 distance between w(lambda=0) and w(lambda=0.9), i.e. without the lambda=1 end.
  c) mean max weight at lambda=0.5, and share of weeks with max weight > 0.9.

Output: results/<tag>/posthoc_front_diagnostics.csv
Usage:  python posthoc_front_diagnostics.py [--synthetic]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-3


def n_distinct(Wl: np.ndarray, tol: float = TOL) -> int:
    """Wl: (n_lambda, d). Greedy count of portfolios more than `tol` (L1) apart."""
    reps = []
    for w in Wl:
        if all(np.abs(w - r).sum() > tol for r in reps):
            reps.append(w)
    return len(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    cfg = CFG
    tag = "synthetic" if a.synthetic else "real"
    z = np.load(ROOT / "cache" / tag / f"weights_{cfg.fingerprint()}.npz", allow_pickle=True)
    W, info = z["W"].item(), z["info"].item()
    lams = [round(float(l), 10) for l in info["lambdas"]]
    j0, j09, j05 = lams.index(0.0), lams.index(0.9), lams.index(cfg.lambda_report)

    rows = []
    for m in info["names"]:
        Wm = W[m]                                     # (T, n_lambda, d)
        if np.isnan(Wm).any():
            raise ValueError(f"{m}: NaN weights in cache")
        nd = np.array([n_distinct(Wm[t]) for t in range(Wm.shape[0])])
        l1 = np.abs(Wm[:, j0] - Wm[:, j09]).sum(axis=1)
        mx = Wm[:, j05].max(axis=1)
        rows.append({"label": "post-hoc", "model": m, "weeks": int(Wm.shape[0]),
                     "distinct_portfolios_mean": nd.mean(),
                     "distinct_portfolios_median": float(np.median(nd)),
                     "L1_w(lam0)_w(lam0.9)_mean": l1.mean(),
                     "max_weight_lam0.5_mean": mx.mean(),
                     "share_weeks_max_weight_gt_0.9": float((mx > 0.9).mean())})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / tag / "posthoc_front_diagnostics.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print("POST-HOC front diagnostics (not pre-registered)")
    print(df.drop(columns="label").round(4).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
