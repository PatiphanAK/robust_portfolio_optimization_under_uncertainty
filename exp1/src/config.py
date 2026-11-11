"""Experiment 1 configuration.

Every setting that could influence results is fixed here BEFORE the out-of-sample
run (pre-registration). Commit this file to git before running on real data and
report the commit hash with the results. Do not edit after looking at test results.
"""
from dataclasses import dataclass, field, asdict
import hashlib
import json


@dataclass(frozen=True)
class Config:
    # ---------------- data ----------------
    # 9 Select Sector SPDR ETFs, one per GICS sector, all trading since Dec-1998.
    # XLRE (2015) and XLC (2018) are excluded because their history is too short.
    tickers: tuple = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
    start: str = "1999-01-01"
    end: str = "2026-10-30"           # last Friday used; fixed in advance
    week_rule: str = "W-FRI"          # last trading day of each week

    # ---------------- forecasting ----------------
    train_window: int = 260           # weeks of history used at each decision
    refit_every: int = 4              # re-estimate AR-GARCH-t params every k weeks
    scale: float = 100.0              # returns are fitted in percent for numerical stability

    # ---------------- risk / objectives ----------------
    alpha: float = 0.05               # CVaR tail probability
    lambdas: tuple = tuple(round(0.1 * i, 1) for i in range(11))
    lambda_report: float = 0.5        # portfolio used for Sharpe / MDD tables
    ridge: float = 1e-6               # tiny ||w||^2 term for unique solutions

    # ---------------- Wasserstein ----------------
    eps_scales: tuple = (0.2, 1.0)            # main results (slide)
    eps_scales_appendix: tuple = (0.5, 2.0)   # reported in appendix, never cherry-picked
    eps_bootstrap_B: int = 100                # bootstrap replicates for base radius
    eps_quantile: float = 0.95
    eps_mean_block: float = 4.0               # stationary bootstrap mean block length (weeks)

    # ---------------- Possibility ----------------
    # B: nested empirical alpha-cuts on FHS scenarios ranked by Mahalanobis distance
    poss_gammas: tuple = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05..0.95
    poss_mc_draws: int = 400_000              # Monte Carlo draws for kappa in method A

    # ---------------- evaluation ----------------
    cost_bp: float = 10.0                     # transaction cost scenario (one-way, bp)
    periods_per_year: int = 52
    hv_ref: tuple = (1.1, 1.1)                # reference point in min-max normalised space
    hv_boot_B: int = 500                      # block bootstrap reps for HV difference CI
    hv_boot_block: float = 8.0
    # market regimes defined in advance (inclusive dates)
    regimes: tuple = (
        ("GFC", "2007-10-01", "2009-03-31"),
        ("EuroDebt", "2011-07-01", "2011-12-31"),
        ("COVID", "2020-02-01", "2020-06-30"),
        ("Bear2022", "2022-01-01", "2022-10-31"),
    )
    seed: int = 20261112

    def as_dict(self):
        return asdict(self)

    def fingerprint(self) -> str:
        s = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:10]


CFG = Config()
