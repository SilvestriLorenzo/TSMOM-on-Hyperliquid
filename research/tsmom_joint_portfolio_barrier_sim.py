"""
H_TSMOM joint-portfolio barrier sim (session 21, user-prompted: "now we are running
multiple combos at the same time").

Every prior p_pass/$-per-month number in this project (tsmom_deployment_config.py,
tsmom_funded_phase_sim.py) simulates ONE combo alone, using the FULL account size, with
its own private +9%/-3% barrier. That is not what's actually running on the live Beta
account: `tsmom_beta_live.py` splits `availableBalance` evenly across all 4 locked
combos (BTC 4h, ETH 8h, ETH 12h, SOL 12h) simultaneously, so in reality there is ONE
account, ONE pooled equity curve, ONE pair of barriers -- fed by 4 quarter-sized,
imperfectly-correlated return streams instead of one full-sized one. Nothing in the repo
had simulated that pooled structure before this script.

Method: reuse each combo's own already-validated per-bar strategy-return series
(`tsmom_walkforward._strat_ret_series`, same signal/ATR-stop/vol-target logic as
production, restricted to that combo's own walk-forward OOS test slice -- never
touching train-period bars, same discipline as every other p_pass number here),
scale by its locked leverage k (session 13/14 table) and by 0.25 (quarter-account
weight, matching `tsmom_beta_live.py`'s actual allocation), then resample to DAILY
log-return sums per combo. The 4 daily series are inner-joined on calendar date --
this is the whole point of the exercise: joint block-bootstrap draws the SAME random
day-blocks for all 4 series at once, preserving real historical same-day co-movement
(the BTC/ETH/SOL intraday co-movement this whole investigation started from), instead
of the independence the single-combo sims implicitly assumed by only ever running one
at a time. Combined daily return = sum of the 4 weighted daily series; barriers
(static -3%, daily -3%, target +9%) are checked on that one pooled path.

Caveats, stated plainly:
- Common calendar overlap across all 4 combos' OOS test slices is ~250 days (BTC 4h's
  test slice starts latest, 2025-12-25) -- the joint correlation structure is estimated
  from a shorter window than any individual combo's own p_pass calculation used.
- Daily resolution loses intraday barrier-breach timing (same caveat H_STRUCT already
  flagged for its own daily-resolution baseline) -- likely slightly overstates p_pass
  for the same reason.
- Quarter-weighting (0.25 each) is `tsmom_beta_live.py`'s choice for paper-trader
  comparability, not an optimized allocation (flagged out-of-scope there too) -- this
  script answers "what does the CURRENT split actually deliver," not "what's optimal."
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_walkforward import walk_forward_one, _strat_ret_series
from tsmom_barrier_sim import TARGET, STATIC_FLOOR, DAILY_LOSS

LOCKED = {
    ("BTC", "4h"): (1, 0.479), ("ETH", "8h"): (1, 0.488),
    ("ETH", "12h"): (1, 0.479), ("SOL", "12h"): (2, 0.437),
}
WEIGHT = 0.25  # tsmom_beta_live.py: availableBalance / 4 (default equal split)
ACCOUNT_SIZE = 100_000  # matches the live Beta paper account tier
EVAL_FEE = 450  # Turbo 1-Step, $100K tier, PROPR.md SS2
BLOCK_LEN_DAYS = 20
SIM_YEARS = 3
N_SIMS = 2000
CYCLE_CAP_DAYS = 1095


def build_daily_series(weights: dict[tuple, float] | None = None) -> pd.DataFrame:
    """weights: {(coin, interval): weight}, must sum to 1.0 (fraction of account
    notional-equivalent allocated to that combo). None = equal quarter-split,
    matching tsmom_beta_live.py's actual default."""
    if weights is None:
        weights = {combo: WEIGHT for combo in LOCKED}
    cols = {}
    for (coin, interval), (lb, k) in LOCKED.items():
        wf = walk_forward_one(coin, interval)
        s = _strat_ret_series(coin, interval, lb)
        test = s.iloc[wf["bar_range"][0]:wf["bar_range"][1]]
        daily = test.resample("1D").sum() * k * weights[(coin, interval)]
        cols[f"{coin}_{interval}"] = daily
    df = pd.DataFrame(cols).dropna(how="any")  # inner join on calendar date
    return df


def joint_block_bootstrap(daily: np.ndarray, n_days: int, block_len: int,
                            rng: np.random.Generator) -> np.ndarray:
    """daily: (n_hist, n_combos) matrix of weighted per-combo daily log returns.
    Draws the SAME random day-blocks across all combos at once -> combined path
    preserves real historical cross-combo correlation, unlike independent per-combo
    bootstraps."""
    n_hist = len(daily)
    out = []
    covered = 0
    while covered < n_days:
        start = rng.integers(0, n_hist)
        idx = (start + np.arange(block_len)) % n_hist
        out.append(daily[idx])
        covered += block_len
    return np.concatenate(out, axis=0)[:n_days].sum(axis=1)  # sum across combos -> pooled daily return


def simulate_pooled_path(daily: np.ndarray, block_len: int, cap_days: int,
                           rng: np.random.Generator) -> tuple[str, int]:
    combined = joint_block_bootstrap(daily, cap_days, block_len, rng)
    equity = 1.0
    for t in range(cap_days):
        prev_equity = equity
        equity *= np.exp(combined[t])
        if equity <= 1 + STATIC_FLOOR:
            return "fail_static", t + 1
        if equity <= prev_equity * (1 - DAILY_LOSS):
            return "fail_daily", t + 1
        if equity >= 1 + TARGET:
            return "pass", t + 1
    return "unresolved", cap_days


def simulate_renewal_pooled(daily: np.ndarray, block_len: int, total_days: float,
                              rng: np.random.Generator) -> dict:
    days_elapsed, net_cash, n_pass, n_fail = 0.0, 0.0, 0, 0
    while days_elapsed < total_days:
        outcome, days = simulate_pooled_path(daily, block_len, CYCLE_CAP_DAYS, rng)
        days_elapsed += days
        if outcome == "pass":
            net_cash += ACCOUNT_SIZE * TARGET * 0.80
            n_pass += 1
        elif outcome in ("fail_static", "fail_daily"):
            net_cash -= EVAL_FEE
            n_fail += 1
        else:
            break
    return {"net_cash": net_cash, "n_pass": n_pass, "n_fail": n_fail, "days": days_elapsed}


if __name__ == "__main__":
    daily_df = build_daily_series()
    print(f"Common calendar overlap across all 4 combos' OOS test slices: "
          f"{daily_df.index.min().date()} -> {daily_df.index.max().date()}, n={len(daily_df)} days")
    print(f"\nPairwise correlation of quarter-weighted daily strategy returns:")
    print(daily_df.corr().round(2).to_string())

    combined_hist = daily_df.sum(axis=1)
    print(f"\nPooled (equal-weight-quarter) daily strategy return: mean={combined_hist.mean():.5f}, "
          f"std={combined_hist.std():.5f}, ann. Sharpe={combined_hist.mean()/combined_hist.std()*np.sqrt(365):.2f}")
    print(f"For reference, mean of each combo's own (unweighted-sum) ann. Sharpe would need re-deriving "
          f"per-combo -- this pooled Sharpe already reflects the correlation/diversification effect.")

    daily_mat = daily_df.to_numpy()
    rng = np.random.default_rng(20260907)

    print(f"\n=== Joint barrier sim: pooled $100K account, {N_SIMS} paths, "
          f"block={BLOCK_LEN_DAYS}d, cap={CYCLE_CAP_DAYS}d ===")
    outcomes = [simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng) for _ in range(N_SIMS)]
    outcome_names = [o[0] for o in outcomes]
    days_to_res = np.array([o[1] for o in outcomes if o[0] != "unresolved"])
    for name in ["pass", "fail_static", "fail_daily", "unresolved"]:
        frac = outcome_names.count(name) / N_SIMS
        print(f"  {name:>12}: {frac:.1%}")
    print(f"  median days to resolution: {np.median(days_to_res):.0f}")
    print(f"  p5-p95 days to resolution: {np.percentile(days_to_res,5):.0f}-{np.percentile(days_to_res,95):.0f}")

    print(f"\n=== Renewal sim (sweep-on-pass policy), {N_SIMS} runs x {SIM_YEARS}y, $100K pooled account ===")
    rng2 = np.random.default_rng(20260908)
    results = [simulate_renewal_pooled(daily_mat, BLOCK_LEN_DAYS, SIM_YEARS * 365, rng2) for _ in range(N_SIMS)]
    per_month = np.array([r["net_cash"] / (r["days"] / 30.44) for r in results])
    passes_per_yr = np.mean([r["n_pass"] / (r["days"] / 365) for r in results])
    fails_per_yr = np.mean([r["n_fail"] / (r["days"] / 365) for r in results])
    print(f"  $/mo mean={per_month.mean():.0f}, median={np.median(per_month):.0f}, "
          f"p5={np.percentile(per_month,5):.0f}, p95={np.percentile(per_month,95):.0f}")
    print(f"  P(net<0 over {SIM_YEARS}y)={np.mean(per_month<0):.1%}, "
          f"passes/yr={passes_per_yr:.2f}, fails/yr={fails_per_yr:.2f}")
