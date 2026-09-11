"""
H_LEVERAGE_SPEED: trade pass-probability for CALENDAR-TIME-AT-RISK, not $/month (session
28 continued, user reframe #2). H_PARTIAL_SCALEOUT answered "does smoothing the trade
distribution accelerate the portfolio" with a clear no ($/month worse at every scale-out
setting tested, because slower resolution outweighs the higher pass rate). The user's
actual concern is different from $/month optimization: the biggest risks on the
currently-deployed baseline are (1) an ~83-day median cycle that resolves negative, or
(2) resolving positive then immediately losing the next ~83-day cycle -- and, more
fundamentally, that market regime can change out from under the strategy during a long
unresolved cycle, plus non-strategy tail risks (platform/counterparty, hacks, regulatory
action) that accumulate with calendar time exposed. The ask: a variant that resolves in
7/10/30 days, even at a LOWER p_pass, so multiple independent attempts fit inside the
same calendar window and each individual exposure window is short -- the opposite
direction from H_PARTIAL_SCALEOUT (which only makes cycles slower).

Method: leverage is the natural dial for calendar-time-to-resolution, independent of the
scale-out mechanism -- scaling the SAME per-combo return stream up moves both the +9%
target and -3%/daily floors closer in calendar-time terms. This reuses the exact overlay
concept already established in this project (session 10-14, tsmom_barrier_sim.py's own
MI&A-calibrated leverage multiplier k applied on top of per-bar vol-targeted sizing) --
here as a simple post-hoc rescale of the already-computed OOS daily-return series
(consistent with how every pooled/joint script in this project treats a leverage
multiplier as a linear scalar on the log-return-scaled per-bar/per-day quantities). No
change to the entry/exit signal itself -- this is orthogonal to H_PARTIAL_SCALEOUT and
could in principle be combined with it later.

Caveat, stated plainly: this rescale does NOT re-check the underlying per-bar
LEVERAGE_CAP (BTC/ETH 5x, SOL 2x) bar-by-bar -- it linearly scales the OOS return
series already realized under the LOCKED k. Headroom check: LOCKED k's are 0.437-0.488,
well below the raw per-asset caps (cap/locked_k ranges from ~10.2x for BTC/ETH to ~4.6x
for SOL, its tightest combo) -- so every k_extra tested here (up to 4x) keeps SOL's
effective multiplier (up to ~1.75) under its 2x cap and BTC/ETH's (up to ~1.95) far under
their 5x cap. This is a plausibility check on the DIAL, not a live-deployable design --
a real implementation would need this re-verified bar-by-bar with the underlying signal
re-run at the higher size, not assumed from the linear rescale used here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_walkforward import walk_forward_one
from tsmom_joint_portfolio_barrier_sim import (
    LOCKED, BLOCK_LEN_DAYS, N_SIMS, CYCLE_CAP_DAYS, SIM_YEARS,
    simulate_pooled_path, simulate_renewal_pooled, build_daily_series, ACCOUNT_SIZE, EVAL_FEE,
)
from tsmom_barrier_sim import TARGET

OUT = Path(__file__).parent / "output"
LEVERAGE_GRID = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
HORIZON_GRID_DAYS = [7, 10, 14, 21, 30, 60, 90]


def resolution_within(days_to_res: np.ndarray, outcome_names: list, horizons: list) -> dict:
    """For each horizon T (calendar days), P(resolved at all within T) and
    P(passed within T) -- the two numbers that directly answer "if I want an answer
    within T days, how likely am I to get one, and how likely is it to be a pass"."""
    out = {}
    days_arr = np.array(days_to_res)
    outcomes_arr = np.array(outcome_names)
    n = len(outcomes_arr)
    for T in horizons:
        resolved_by_T = (outcomes_arr != "unresolved") & (days_arr <= T)
        out[f"p_resolved_le_{T}d"] = resolved_by_T.mean()
        out[f"p_pass_le_{T}d"] = ((outcomes_arr == "pass") & (days_arr <= T)).mean()
    return out


if __name__ == "__main__":
    wf_cache = {combo: walk_forward_one(*combo) for combo in LOCKED}
    baseline_df = build_daily_series()
    baseline_mat = baseline_df.to_numpy()

    print(f"Headroom check: LOCKED k's = {[round(v[1], 3) for v in LOCKED.values()]}, "
          f"caps = BTC/ETH 5.0x, SOL 2.0x -- effective multiplier at k_extra=4.0: "
          f"{[round(v[1] * 4.0, 2) for v in LOCKED.values()]}")

    rows = []
    for k_extra in LEVERAGE_GRID:
        daily_mat = baseline_mat * k_extra
        combined_hist = daily_mat.sum(axis=1)
        sharpe = combined_hist.mean() / combined_hist.std() * np.sqrt(365) if combined_hist.std() > 0 else np.nan

        rng = np.random.default_rng(20260907)
        outcomes = [simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng) for _ in range(N_SIMS)]
        outcome_names = [o[0] for o in outcomes]
        days_to_res = [o[1] for o in outcomes]  # unresolved paths carry cap_days -- fine, they're > every horizon tested

        row = {"k_extra": k_extra, "sharpe": sharpe}
        for oc in ["pass", "fail_static", "fail_daily", "unresolved"]:
            row[oc] = outcome_names.count(oc) / N_SIMS
        resolved_days = [d for d, o in zip(days_to_res, outcome_names) if o != "unresolved"]
        row["median_days_all"] = float(np.median(resolved_days)) if resolved_days else float("nan")
        row.update(resolution_within(days_to_res, outcome_names, HORIZON_GRID_DAYS))

        rng2 = np.random.default_rng(20260908)
        renewals = [simulate_renewal_pooled(daily_mat, BLOCK_LEN_DAYS, SIM_YEARS * 365, rng2) for _ in range(N_SIMS)]
        per_month = np.array([r["net_cash"] / (r["days"] / 30.44) for r in renewals])
        row["dollars_per_month_mean"] = per_month.mean()
        row["cycles_per_yr"] = np.mean([(r["n_pass"] + r["n_fail"]) / (r["days"] / 365) for r in renewals])

        rows.append(row)
        print(f"  done: k_extra={k_extra}")

    df = pd.DataFrame(rows).set_index("k_extra")
    print("\n=== Leverage sweep, pooled $100K account (k_extra=1.0 == current baseline) ===")
    fmt = {c: "{:.1%}".format for c in ["pass", "fail_static", "fail_daily", "unresolved"]
           if c in df.columns}
    fmt.update({c: "{:.1%}".format for c in df.columns if c.startswith("p_resolved") or c.startswith("p_pass_le")})
    fmt["sharpe"] = "{:.2f}".format
    fmt["median_days_all"] = "{:.0f}".format
    fmt["dollars_per_month_mean"] = "${:,.0f}".format
    fmt["cycles_per_yr"] = "{:.2f}".format

    print(df[["sharpe", "pass", "fail_static", "median_days_all", "cycles_per_yr", "dollars_per_month_mean"]]
          .to_string(formatters=fmt))

    print("\n=== Resolved-within-T-days probabilities ===")
    horizon_cols_resolved = [f"p_resolved_le_{T}d" for T in HORIZON_GRID_DAYS]
    print(df[horizon_cols_resolved].to_string(formatters=fmt))

    print("\n=== Pass-within-T-days probabilities ===")
    horizon_cols_pass = [f"p_pass_le_{T}d" for T in HORIZON_GRID_DAYS]
    print(df[horizon_cols_pass].to_string(formatters=fmt))

    OUT.mkdir(exist_ok=True)
    df.reset_index().to_json(OUT / "tsmom_leverage_speed_sweep.json", orient="records", indent=2)
