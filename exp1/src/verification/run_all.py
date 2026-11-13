"""Algorithm verification suite. Run from src/:  python -m verification.run_all
(or:  python verification/run_all.py).  Exit code 0 iff every check passes.

  check_optimizer : global optimality, Pareto monotonicity, Poss-A formula, ridge
  check_robust    : Wasserstein tightness/upper bound, possibility cuts, kappas
  check_backtest  : return timing, turnover/cost by hand, mini backtest + look-ahead
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, fixture
import check_optimizer, check_robust, check_backtest


def main():
    ck, t0 = Checker(), time.time()
    fx = fixture()
    for mod in (check_optimizer, check_robust, check_backtest):
        print(f"\n########## {mod.__name__}")
        mod.run(ck, fx)
    print(f"\n{ck.n_pass} passed, {ck.n_fail} failed in {time.time() - t0:.0f}s")
    print("ALL PASS" if ck.ok else "SOME CHECKS FAILED")
    return 0 if ck.ok else 1


if __name__ == "__main__":
    sys.exit(main())
