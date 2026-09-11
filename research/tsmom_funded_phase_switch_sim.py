"""
H_FUNDED_PHASE_SWITCH (session 28 continued): the user's concrete, immediate plan --
buy a real $10K Turbo attempt tomorrow (discounted first fee $35) running the
H_LEVERAGE_SPEED k_extra=2.0 variant. Two questions once funded: how much can the
funded account actually make, and should leverage drop back to baseline immediately on
funding or run hot a little longer first.

Funded-phase mechanic, per the user's own description (more authoritative than this
repo's own prior guess in tsmom_funded_phase_sim.py's docstring, which flagged this
exact point as unconfirmed): once funded, there is NO required profit target -- payout
is requestable any time once profit clears $20, same static (3%) and daily (3%) loss
floors apply continuously, and a floor breach revokes the funded slot (re-entry needs a
fresh eval purchase). This still structurally matches the existing renewal-cycle
machinery (`simulate_pooled_path`/`simulate_renewal_pooled`) IF the trader's own chosen
payout habit is modeled as "sweep whenever equity first reaches +T%, resetting the
account to its starting balance" -- exactly analogous to the eval's own +9%/-3% cycle,
just with T chosen freely rather than fixed at 9%. This script keeps T=9% (TARGET) as
the primary, most directly comparable policy (reuses `simulate_pooled_path` as-is) and
flags -- but does not fully grid-search -- that a SMALLER T (sweep smaller gains more
often) is a separate, real lever the user could also pull, structurally the same
speed-vs-size tradeoff as H_LEVERAGE_SPEED itself, just applied to the payout threshold
instead of the eval target.

Three policies compared, at the real $10K tier (first-buy fee $35 as discounted, $50 for
any subsequent re-buy after a funded-phase breach -- assumed NOT discounted again,
flagged as an assumption):
  - stay_2x:      keep k_extra=2.0 forever (post-funding leverage never drops)
  - immediate_1x: drop to baseline (k_extra=1.0) the moment the funding pass happens
  - wait_1_cycle: stay at k_extra=2.0 for exactly one more completed cycle after the
                  funding pass (the user's literal "wait a little more" option), then
                  drop to baseline for good
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_walkforward import walk_forward_one
from tsmom_joint_portfolio_barrier_sim import LOCKED, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, TARGET, build_daily_series
from tsmom_leverage_speed_capped import build_daily_series_leveraged
from tsmom_joint_portfolio_barrier_sim import simulate_pooled_path

OUT = Path(__file__).parent / "output"
SPLIT = 0.80
SIM_YEARS = 1
N_SIMS = 3000
POLICIES = ["stay_2x", "immediate_1x", "wait_1_cycle"]

# Tier: (account_size, first_fee (with the user's stated discount), restart_fee (full
# price -- assumed the discount does NOT reapply on a re-buy after a funded-phase
# breach; flag/revisit if a repeat discount is actually available)
TIERS = {
    "10K": (10_000, 35, 50),      # 30% off $50
    "25K": (25_000, 87.5, 125),   # 30% off $125
    "50K": (50_000, 245, 245),    # no stated discount -- full price both first buy and restart
}


def simulate_funded(account_size: float, first_fee: float, restart_fee: float,
                     daily_2x: np.ndarray, daily_1x: np.ndarray, policy: str,
                     total_days: float, rng: np.random.Generator) -> dict:
    """Renewal loop identical in structure to simulate_renewal_pooled, except the FIRST
    cycle always uses the leveraged signal (the eval attempt itself), and which signal
    subsequent cycles use depends on `policy` and how many passes have occurred since
    the funding pass. A breach at any point costs `restart_fee` and resumes at
    k_extra=2.0 (a fresh eval attempt, same as the original plan) regardless of policy."""
    days_elapsed, net_cash, n_pass, n_fail, n_pass_since_funding = 0.0, -first_fee, 0, 0, 0
    funded = False
    while days_elapsed < total_days:
        if not funded:
            daily_mat = daily_2x  # the eval attempt itself is always run at k_extra=2.0
        elif policy == "stay_2x":
            daily_mat = daily_2x
        elif policy == "immediate_1x":
            daily_mat = daily_1x
        elif policy == "wait_1_cycle":
            daily_mat = daily_2x if n_pass_since_funding < 1 else daily_1x
        else:
            raise ValueError(policy)

        outcome, days = simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng)
        days_elapsed += days
        if outcome == "pass":
            net_cash += account_size * TARGET * SPLIT
            n_pass += 1
            if funded:
                n_pass_since_funding += 1
            funded = True
        elif outcome in ("fail_static", "fail_daily"):
            net_cash -= restart_fee
            n_fail += 1
            funded = False
            n_pass_since_funding = 0
        else:
            break
    return {"net_cash": net_cash, "n_pass": n_pass, "n_fail": n_fail, "days": days_elapsed, "ever_funded": n_pass > 0}


if __name__ == "__main__":
    wf_cache = {combo: walk_forward_one(*combo) for combo in LOCKED}
    daily_2x = build_daily_series_leveraged(2.0, wf_cache)[0].to_numpy()
    daily_1x = build_daily_series().to_numpy()
    total_days = SIM_YEARS * 365

    rows = []
    for tier_name, (account_size, first_fee, restart_fee) in TIERS.items():
        for policy in POLICIES:
            rng = np.random.default_rng(20260910)
            results = [simulate_funded(account_size, first_fee, restart_fee, daily_2x, daily_1x,
                                        policy, total_days, rng) for _ in range(N_SIMS)]
            per_month = np.array([r["net_cash"] / (r["days"] / 30.44) for r in results])
            n_fail_arr = np.array([r["n_fail"] for r in results])
            n_pass_arr = np.array([r["n_pass"] for r in results])
            rows.append({
                "tier": tier_name, "policy": policy,
                "dollars_per_month_mean": per_month.mean(),
                "dollars_per_month_median": np.median(per_month),
                "dollars_per_year_mean": per_month.mean() * 12,
                "p_net_negative_1y": np.mean(per_month < 0),
                "breaches_per_yr_mean": n_fail_arr.mean(),
                "passes_per_yr_mean": n_pass_arr.mean(),
                "p_never_funded_1y": np.mean([not r["ever_funded"] for r in results]),
            })
            print(f"  done: {tier_name} / {policy}")

    df = pd.DataFrame(rows).set_index(["tier", "policy"])
    fmt = {"dollars_per_month_mean": "${:,.0f}".format, "dollars_per_month_median": "${:,.0f}".format,
           "dollars_per_year_mean": "${:,.0f}".format, "p_net_negative_1y": "{:.1%}".format,
           "breaches_per_yr_mean": "{:.2f}".format, "passes_per_yr_mean": "{:.2f}".format,
           "p_never_funded_1y": "{:.2%}".format}
    print(f"\n=== Funded-phase policy comparison across tiers, 1-year horizon, {N_SIMS} sims/cell ===")
    print(df.to_string(formatters=fmt))

    OUT.mkdir(exist_ok=True)
    df.reset_index().to_json(OUT / "tsmom_funded_phase_switch_sim.json", orient="records", indent=2)
