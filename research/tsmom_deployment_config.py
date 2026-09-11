"""
H_TSMOM deployment-config selection (session 13, numbers corrected session 14 after
tsmom_barrier_sim.py's constant-vs-vol-targeted sizing fix -- see that module's
docstring), continuing tsmom_walkforward.py / tsmom_sizing_retune.py. Purpose: turn the
research findings so far into ONE fixed, written-down config per candidate combo -- not
a number re-optimized after the fact.
written-down config per candidate combo -- not a number re-optimized after the fact.

Two decisions get made here, both DELIBERATELY NOT by searching for the best cell on
the same held-out test window sizing_retune.py already used (that grid's argmax was
explicitly flagged there as likely optimistic -- picking a fresh argmax for "the real
deployment number" would just repeat the same in-sample-selection mistake one more
time):

1. MI&A target_mdd: fixed at Propr's own full static-drawdown budget, 3% -- a number
   that comes directly from PROPR.md's rulebook, not from a search.
2. MI&A calibration horizon: fixed at the MEASURED median bars-to-resolution of the
   already-validated fixed-vol baseline (no sizing search at all -- just observing how
   fast the strategy naturally resolves), converted to days. This replaces both the
   original session-10 "1095 days" convention (shown to be far too long -- most paths
   resolve long before that) and sizing_retune.py's swept horizon grid (each cell an
   arbitrary guess) with a number actually measured from the strategy's own behavior.

Combos evaluated: the four that came out of the walk-forward + sizing-retune work as
strongest on the combination of (a) out-of-sample p_pass, (b) train-selected lookback
matching the test-optimal lookback (a parameter-stability signal), and (c) how much of
the MI&A sizing grid beat baseline (a robustness signal, not just the single best cell):
BTC 4h, ETH 8h, ETH 12h, SOL 12h (SOL 12h was one of session 10's original three but
scores weakest on (c) -- kept here for an explicit side-by-side, not assumed excluded).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_walkforward import walk_forward_one
from tsmom_barrier_sim import simulate_path, run_barrier_sim

CANDIDATES = [("BTC", "4h"), ("ETH", "8h"), ("ETH", "12h"), ("SOL", "12h")]
STRUCTURAL_TARGET_MDD = 0.03  # Propr's full static drawdown budget, PROPR.md Section 4
N_PATHS_RESOLUTION = 400
N_PATHS_FINAL = 800
BARS_PER_DAY_BY_INTERVAL = {"4h": 6, "8h": 3, "12h": 2}

# Previously-found argmax-searched "best" cell per combo (tsmom_sizing_retune.py output),
# reported here ONLY as a reference point for how much the structural (non-searched)
# choice below leaves on the table -- NOT used to pick anything.
PRIOR_ARGMAX_P_PASS = {
    ("BTC", "4h"): 0.380, ("ETH", "8h"): 0.462, ("ETH", "12h"): 0.375, ("SOL", "12h"): 0.275,
}


def median_bars_to_resolution(coin: str, interval: str, lb_star: int, bar_range: tuple[int, int],
                               leverage: float, n_paths: int = N_PATHS_RESOLUTION) -> int:
    """Measures how fast the ALREADY-VALIDATED fixed-vol baseline naturally resolves
    (pass or fail, whichever comes first) -- no sizing search involved, just observing
    the strategy's own pace, to ground MI&A's calibration horizon in something real."""
    from tsmom_barrier_sim import extract_market_tuples
    log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval, bar_range)
    rng = np.random.default_rng(seed=1)
    n_bars = int(1095 * 24 / int(interval[:-1]))  # long enough ceiling; median resolves well before this
    bars_list = []
    for _ in range(n_paths):
        outcome, bars = simulate_path(coin, interval, lb_star, 14, 2.5, leverage, cppi=False,
                                       n_bars=n_bars, log_ret_hist=log_ret_hist, hi_hist=hi_hist,
                                       lo_hist=lo_hist, rng=rng, return_bars=True)
        if outcome != "unresolved":
            bars_list.append(bars)
    median_bars = int(np.median(bars_list)) if bars_list else n_bars
    bars_per_day = BARS_PER_DAY_BY_INTERVAL[interval]
    return max(1, round(median_bars / bars_per_day))


if __name__ == "__main__":
    print(f"{'coin':>4} {'interval':>8} | {'lb*':>3} | {'flat p_pass':>11} | "
          f"{'median days':>11} | {'MI&A@3%/measured p_pass':>24} | {'prior argmax p_pass':>20}")
    rows = []
    for coin, interval in CANDIDATES:
        wf = walk_forward_one(coin, interval)
        median_days = median_bars_to_resolution(coin, interval, wf["lb_star"], wf["bar_range"], 1.0)
        sim = run_barrier_sim(coin, interval, wf["lb_star"], n_paths=N_PATHS_FINAL, cppi=False,
                               leverage=None, bar_range=wf["bar_range"],
                               mia_target_mdd=STRUCTURAL_TARGET_MDD, mia_horizon_days=median_days)
        rows.append({"coin": coin, "interval": interval, "lb_star": wf["lb_star"],
                     "flat_p_pass": wf["oos_p_pass"], "median_days_to_resolution": median_days,
                     "mia_leverage": sim["leverage"], "mia_p_pass": sim["pass"],
                     "mia_fail_static": sim["fail_static"], "mia_fail_daily": sim["fail_daily"],
                     "mia_unresolved": sim["unresolved"],
                     "prior_argmax_p_pass": PRIOR_ARGMAX_P_PASS[(coin, interval)]})
        r = rows[-1]
        print(f"{coin:>4} {interval:>8} | {wf['lb_star']:>3} | {wf['oos_p_pass']:>10.1%} | "
              f"{median_days:>10}d | lev={sim['leverage']:.3f} -> {sim['pass']:>10.1%}       | "
              f"{PRIOR_ARGMAX_P_PASS[(coin, interval)]:>19.1%}")

    print("\nNote: 'prior argmax p_pass' is the sizing_retune.py grid's single best cell for "
          "reference only -- it was explicitly flagged there as likely optimistic (argmax over "
          "16 cells scored on the same held-out window). The structural (target=3% full budget, "
          "horizon=measured median resolution time) column is what this session actually locks in.")
