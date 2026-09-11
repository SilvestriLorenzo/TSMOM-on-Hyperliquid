"""
H_BUDGET_RUIN -- personal-savings ruin simulation for the real-money $5K Turbo
sequencing question (user, session 2026-09-08): if repeated real fails have to be paid
out of personal savings while a pass is self-funding, what's the actual probability of
burning through the whole budget before ever landing a pass, and how many fails /
how much calendar time should realistically be expected first?

Reuses tsmom_joint_portfolio_barrier_sim's exact pooled-path bootstrap unmodified (same
4 locked combos, same quarter-weighting as tsmom_beta_live.py, same block length/cap) at
$5K account size instead of $100K -- p_pass and cycle-length are account-size-invariant
(percentage-based rules, established repeatedly elsewhere in this project), only the $
payout changes. Reports both the full-250-day-sample number and the ex-Aug-17-27-rally
number side by side, same "don't trust the headline before excision" discipline as
H_JOINT_PORTFOLIO itself.

Caveat carried over unchanged: this measures "given the historically-observed p_pass
distribution (bootstrap-resampled from ~250 days, mostly a trending 2025-2026 regime)
holds going forward" -- it is NOT a claim that the real live edge can't structurally
degrade below what's simulated here. See HYPOTHESES.md's H_TSMOM funded-phase-economics
"decay risk, honestly assessed, not modeled" note.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_joint_portfolio_barrier_sim import build_daily_series, simulate_pooled_path, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS

ACCOUNT_SIZE = 5_000
EVAL_FEE = 25  # Turbo 1-Step, $5K tier, PROPR.md SS2
TARGET = 0.09
SPLIT = 0.80
BUDGET = 637  # ~590 EUR at ~1.08 EUR/USD -- rerun with the real fee/FX if it moves
N_SIMS = 4000
EPISODE_START, EPISODE_END = "2026-08-17", "2026-08-27"


def simulate_savings_ruin(daily: np.ndarray, block_len: int, cap_days: int,
                            budget: float, fee: float, rng: np.random.Generator) -> dict:
    spent, n_fail, days_elapsed = 0.0, 0, 0
    while budget - spent >= fee:
        outcome, days = simulate_pooled_path(daily, block_len, cap_days, rng)
        days_elapsed += days
        if outcome == "pass":
            return {"status": "pass", "n_fail": n_fail, "days": days_elapsed, "spent": spent + fee}
        if outcome == "unresolved":
            return {"status": "unresolved", "n_fail": n_fail, "days": days_elapsed, "spent": spent}
        spent += fee
        n_fail += 1
    return {"status": "ruined", "n_fail": n_fail, "days": days_elapsed, "spent": spent}


def report(daily_mat: np.ndarray, label: str) -> None:
    rng = np.random.default_rng(20260909)
    results = [simulate_savings_ruin(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, BUDGET, EVAL_FEE, rng)
               for _ in range(N_SIMS)]
    ruined = np.mean([r["status"] == "ruined" for r in results])
    unresolved = np.mean([r["status"] == "unresolved" for r in results])
    passed = [r for r in results if r["status"] == "pass"]
    fails = np.array([r["n_fail"] for r in passed])
    days = np.array([r["days"] for r in passed])
    spent = np.array([r["spent"] for r in passed])

    print(f"=== {label} ===")
    print(f"P(exhaust ${BUDGET} budget before ever passing) = {ruined:.1%}")
    print(f"P(pass before exhausting budget)                = {1 - ruined - unresolved:.1%}"
          f"{f'  (unresolved: {unresolved:.1%})' if unresolved else ''}")
    if len(passed):
        print(f"Conditional on eventually passing ({len(passed)}/{N_SIMS} sims):")
        print(f"  fails before the pass: mean={fails.mean():.2f}, median={np.median(fails):.0f}, "
              f"p95={np.percentile(fails, 95):.0f}")
        print(f"  $ spent on fees incl. the winning attempt: mean=${spent.mean():.0f}, "
              f"median=${np.median(spent):.0f}, p95=${np.percentile(spent, 95):.0f}")
        print(f"  calendar days to first pass: mean={days.mean():.0f}, median={np.median(days):.0f}, "
              f"p5-p95={np.percentile(days, 5):.0f}-{np.percentile(days, 95):.0f}")
    print()


if __name__ == "__main__":
    daily_df = build_daily_series()
    print(f"Budget ${BUDGET}, fee ${EVAL_FEE}/attempt (max {BUDGET // EVAL_FEE} back-to-back fails "
          f"affordable), {N_SIMS} simulated savers\n")

    report(daily_df.to_numpy(), f"Full {len(daily_df)}-day sample (incl. Aug 17-27 rally)")

    mask = (daily_df.index < EPISODE_START) | (daily_df.index > EPISODE_END)
    ex_episode = daily_df[mask]
    report(ex_episode.to_numpy(), f"Ex Aug 17-27 rally ({len(ex_episode)}/{len(daily_df)} days)")
