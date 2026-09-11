"""
H_BUDGET_RUIN, leverage-speed follow-up (session 28 continued). budget_ruin_sim.py
(session 22, unmodified, reused here) already established the right way to reason about
repeated real-money attempts against the user's own ~$637 budget: SEQUENTIAL re-buys
(fail -> spend the fee -> try again with what's left), not simultaneous parallel
launches -- session 22 separately found that firing multiple accounts at once on the
identical strategy gives near-zero pass/fail diversification (~1 correlation, same
market driving all of them at the same time); real diversification requires staggered
starts, which sequential re-buying already models correctly (each new attempt starts
after the previous one resolved, on a fresh bootstrap draw).

This reruns that exact ruin simulation with the H_LEVERAGE_SPEED variant (k_extra =
2.0/2.5/3.0, properly per-bar-capped per tsmom_leverage_speed_capped.py, confirmed zero
clipping in this range) instead of the baseline signal, at the $5K and $10K Turbo tiers
(fees $25/$50, PROPR.md SS2) -- both to see whether faster-but-lower-p_pass cycles
actually get the user to a first pass with less calendar time and no worse (or better)
ruin risk than baseline, which is the concrete question behind "I don't personally waste
time but calendar days expose us to regime change."
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_walkforward import walk_forward_one
from tsmom_joint_portfolio_barrier_sim import LOCKED, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, build_daily_series
from tsmom_leverage_speed_capped import build_daily_series_leveraged
from budget_ruin_sim import simulate_savings_ruin, BUDGET

OUT = Path(__file__).parent / "output"
N_SIMS = 3000
TIERS = {"5K": 25, "10K": 50}  # PROPR.md SS2 Turbo 1-Step fees
K_EXTRA_GRID = [1.0, 2.0, 2.5, 3.0]


def report(daily_mat: np.ndarray, tier: str, fee: float, k_extra: float) -> dict:
    rng = np.random.default_rng(20260909)  # same seed convention as budget_ruin_sim.py
    results = [simulate_savings_ruin(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, BUDGET, fee, rng)
               for _ in range(N_SIMS)]
    ruined = np.mean([r["status"] == "ruined" for r in results])
    unresolved = np.mean([r["status"] == "unresolved" for r in results])
    passed = [r for r in results if r["status"] == "pass"]
    fails = np.array([r["n_fail"] for r in passed])
    days = np.array([r["days"] for r in passed])
    spent = np.array([r["spent"] for r in passed])

    row = {
        "tier": tier, "k_extra": k_extra, "fee": fee,
        "p_ruined": ruined, "p_pass_before_ruin": 1 - ruined - unresolved,
        "p_unresolved": unresolved, "n_pass_sims": len(passed),
    }
    if len(passed):
        row.update({
            "mean_fails_before_pass": fails.mean(), "median_days_to_pass": float(np.median(days)),
            "p5_days_to_pass": float(np.percentile(days, 5)), "p95_days_to_pass": float(np.percentile(days, 95)),
            "mean_spent_incl_win": spent.mean(),
        })
    return row


if __name__ == "__main__":
    wf_cache = {combo: walk_forward_one(*combo) for combo in LOCKED}
    print(f"Budget ${BUDGET} (~590 EUR), {N_SIMS} simulated savers per cell\n")

    rows = []
    for k_extra in K_EXTRA_GRID:
        daily_mat = (build_daily_series() if k_extra == 1.0
                     else build_daily_series_leveraged(k_extra, wf_cache)[0]).to_numpy()
        for tier, fee in TIERS.items():
            row = report(daily_mat, tier, fee, k_extra)
            rows.append(row)
            print(f"  done: tier=${tier}, k_extra={k_extra}")

    import pandas as pd
    df = pd.DataFrame(rows)
    fmt = {"p_ruined": "{:.1%}".format, "p_pass_before_ruin": "{:.1%}".format,
           "p_unresolved": "{:.1%}".format, "mean_fails_before_pass": "{:.2f}".format,
           "median_days_to_pass": "{:.0f}".format, "p5_days_to_pass": "{:.0f}".format,
           "p95_days_to_pass": "{:.0f}".format, "mean_spent_incl_win": "${:,.0f}".format}
    print("\n=== Sequential real-budget attempts, baseline vs. leverage-speed variants ===")
    print(df.set_index(["tier", "k_extra"])[
        ["p_ruined", "p_pass_before_ruin", "mean_fails_before_pass",
         "median_days_to_pass", "p5_days_to_pass", "p95_days_to_pass", "mean_spent_incl_win"]
    ].to_string(formatters=fmt))

    OUT.mkdir(exist_ok=True)
    df.to_json(OUT / "tsmom_leverage_ruin_sim.json", orient="records", indent=2)
