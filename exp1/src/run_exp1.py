"""Experiment 1: Nominal vs Wasserstein vs Possibility (A: elliptical, B: scenario)
on the same AR(1)-GARCH(1,1)-t forecasts.

Usage
  python run_exp1.py --download        # first run on your machine: fetch Yahoo data
  python run_exp1.py                   # re-run using data/prices_daily.csv
  python run_exp1.py --synthetic       # pipeline test only (numbers are NOT results)
Outputs go to results/<tag>/ : tables (.csv, .tex), figures (.png), run_meta.json
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG
import data as D
from forecast import run_forecasts
from backtest import run_backtest, realized_returns, model_names
import metrics as M

ROOT = Path(__file__).resolve().parent.parent


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "not-a-git-repo"


def fmt_tex(df: pd.DataFrame, path: Path, floatfmt="%.3f"):
    path.write_text(df.to_latex(float_format=lambda x: floatfmt % x, escape=True, na_rep="--"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--weeks", type=int, default=1450, help="synthetic length")
    ap.add_argument("--no-appendix", action="store_true")
    a = ap.parse_args()
    cfg = CFG

    tag = "synthetic" if a.synthetic else "real"
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache" / tag
    cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ data
    if a.synthetic:
        R = D.synthetic_returns(cfg, n_weeks=a.weeks, seed=1)
        data_meta = {"source": "SYNTHETIC - pipeline test only"}
    else:
        # validated prices; a bad cache is repaired, nothing incomplete is used or saved
        px = D.ensure_prices(cfg, refresh=a.download)
        R = D.weekly_log_returns(px, cfg)
        mp = D.DATA_DIR / "prices_daily.meta.json"
        data_meta = json.loads(mp.read_text()) if mp.exists() else {}
    print(f"weekly returns: {R.shape}  {R.index[0].date()} .. {R.index[-1].date()}")

    # ------------------------------------------------------------ forecasts + decisions
    print("[1/3] AR-GARCH-t forecasts")
    fc = run_forecasts(R, cfg, cache=cache / "forecasts.npz")
    print("[2/3] portfolio decisions")
    W, info = run_backtest(R, fc, cfg, cache=cache / f"weights_{cfg.fingerprint()}.npz",
                           appendix=not a.no_appendix)
    print("   bad solves:", info["bad_solves"])

    # ------------------------------------------------------------ evaluation
    print("[3/3] evaluation")
    rets = realized_returns(R, fc, W, cfg)
    names = info["names"] + ["1/N"]
    main_names = [n for n in names if not any(f"x{s:g}" == n.split()[-1] for s in cfg.eps_scales_appendix)]
    lams = list(cfg.lambdas)
    jr = lams.index(cfg.lambda_report)
    edates = pd.DatetimeIndex(info["eval_dates"])

    # summary table at lambda_report
    rows = {}
    for m in names:
        g, n, to = rets[m]["gross"][:, jr], rets[m]["net"][:, jr], rets[m]["turnover"][:, jr]
        s = M.summary(g, to, cfg.alpha, cfg.periods_per_year)
        s["Sharpe (net)"] = M.sharpe(n, cfg.periods_per_year)
        rows[m] = s
    summ = pd.DataFrame(rows).T

    # OOS fronts + HV (gross)
    fronts = {m: M.oos_front(rets[m]["gross"], cfg.alpha) for m in names if m != "1/N"}
    fronts_main = {m: fronts[m] for m in main_names if m != "1/N"}
    hv, (lo, span) = M.normalised_hv(fronts_main, cfg.hv_ref)
    rng = np.random.default_rng(cfg.seed + 1)
    ci = M.hv_diff_ci({m: rets[m]["gross"] for m in fronts_main}, "Nominal", cfg.alpha,
                      cfg.hv_ref, cfg.hv_boot_B, cfg.hv_boot_block, rng)
    summ["OOS HV"] = pd.Series(hv)
    summ["dHV vs Nominal (95% CI)"] = pd.Series(
        {m: f"[{c[0]:+.3f}, {c[1]:+.3f}]" for m, c in ci.items()})

    # Sharpe tests vs 1/N and vs Nominal, Holm-adjusted within each family
    p_1n, p_nom = {}, {}
    for m in names:
        if m == "1/N":
            continue
        _, p_1n[m] = M.sharpe_diff_test(rets[m]["gross"][:, jr], rets["1/N"]["gross"][:, jr])
        if m != "Nominal":
            _, p_nom[m] = M.sharpe_diff_test(rets[m]["gross"][:, jr], rets["Nominal"]["gross"][:, jr])
    summ["p vs 1/N (Holm)"] = pd.Series(M.holm(p_1n))
    summ["p vs Nominal (Holm)"] = pd.Series(M.holm(p_nom))

    summ_main = summ.loc[main_names]
    summ.to_csv(out / "table_summary_all.csv")
    summ_main.to_csv(out / "table_summary_main.csv")
    num_cols = [c for c in summ_main.columns if c != "dHV vs Nominal (95% CI)"]
    perf_cols = ["Ann. return %", "Ann. vol %", "Sharpe", "Sharpe (net)", "MDD %", "CVaR %", "Turnover"]
    front_cols = ["OOS HV", "dHV vs Nominal (95% CI)", "p vs 1/N (Holm)", "p vs Nominal (Holm)"]
    fmt_tex(summ_main[perf_cols], out / "table_perf_main.tex", "%.2f")
    fmt_tex(summ_main[front_cols].drop(index="1/N"), out / "table_front_main.tex")
    fmt_tex(summ[perf_cols], out / "table_perf_all.tex", "%.2f")

    # regimes (gross, lambda_report)
    reg_rows = []
    for name, s, e in cfg.regimes + (("Full", str(edates[0].date()), str(edates[-1].date())),):
        mask = (edates >= s) & (edates <= e)
        if mask.sum() < 8:
            continue
        for m in main_names:
            r = rets[m]["gross"][mask, jr]
            reg_rows.append({"Regime": name, "Weeks": int(mask.sum()), "Model": m,
                             "Sharpe": M.sharpe(r), "CVaR %": 100 * M.cvar_loss(r, cfg.alpha),
                             "MDD %": 100 * M.max_drawdown(r)})
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(out / "table_regimes.csv", index=False)
    if len(reg):
        piv = reg.pivot_table(index="Model", columns="Regime", values="CVaR %").loc[main_names]
        fmt_tex(piv, out / "table_regimes_cvar.tex", "%.2f")
        piv = reg.pivot_table(index="Model", columns="Regime", values="Sharpe").loc[main_names]
        fmt_tex(piv, out / "table_regimes_sharpe.tex", "%.2f")

    # front-length diagnostic (Poss-A collapse): spread of the in-sample weights over lambda
    spread = {m: float(np.mean(np.abs(W[m][:, 0] - W[m][:, -1]).sum(1))) for m in info["names"]}
    pd.Series(spread, name="mean L1(w_l=0 - w_l=1)").to_csv(out / "front_spread.csv")

    # ------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for m in fronts_main:
        P = 100 * fronts_main[m]
        pts, = ax.plot(P[:, 0], P[:, 1], "o", ms=3.5, label=f"{m} (HV {hv[m]:.3f})")
        ND = M.nondominated(P)
        ND = ND[np.argsort(ND[:, 0])]
        ax.plot(ND[:, 0], ND[:, 1], "-", lw=1.2, color=pts.get_color())
    p1n = 100 * M.oos_front(rets["1/N"]["gross"][:, :1], cfg.alpha)[0]
    ax.plot(*p1n, "k*", ms=12, label="1/N")
    ax.set_xlabel("Realised mean weekly loss (%)", fontsize=11)
    ax.set_ylabel(f"Realised CVaR {cfg.alpha:.0%} (%)", fontsize=11)
    ax.set_title("Out-of-sample fronts (lambda = 0 ... 1)")
    ax.legend(fontsize=9)
    fig.text(0.5, 0.01, "points = λ 0…1; line = non-dominated points", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "fig_oos_front.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in main_names:
        ax.plot(edates, np.cumprod(1 + rets[m]["gross"][:, jr]), lw=1.1 if m != "1/N" else 1.8,
                color="k" if m == "1/N" else None, label=m)
    for _, s, e in cfg.regimes:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="0.9", zorder=0)
    ax.set_yscale("log"); ax.set_ylabel("Wealth (log)")
    ax.set_title(f"Out-of-sample wealth, lambda = {cfg.lambda_report}")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "fig_wealth.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(pd.DatetimeIndex(info["dates"]), info["eps0"], lw=1)
    ax.set_ylabel("base radius eps0"); ax.set_title("Wasserstein base radius (bootstrap 95% quantile)")
    fig.tight_layout(); fig.savefig(out / "fig_eps0.png", dpi=200)

    # ------------------------------------------------------------ metadata
    meta = {"config_fingerprint": cfg.fingerprint(), "git_commit": git_commit(),
            "config": cfg.as_dict(), "data": data_meta,
            "weeks": int(R.shape[0]), "decisions": len(fc),
            "oos_first": str(edates[0].date()), "oos_last": str(edates[-1].date()),
            "scenarios_N": info["N"], "bad_solves": info["bad_solves"],
            "hv_norm_lo": lo.tolist(), "hv_norm_span": span.tolist()}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    pd.set_option("display.width", 200)
    print(summ_main[num_cols].round(3))
    print("front spread:", {k: round(v, 3) for k, v in spread.items()})
    print("saved to", out)


if __name__ == "__main__":
    main()
