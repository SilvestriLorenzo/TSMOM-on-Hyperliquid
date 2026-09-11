"""
H_PARTIAL_SCALEOUT, pooled-account follow-up (session 28 continued). The main script's
(tsmom_partial_scaleout_variant.py) median-days-to-resolution numbers use EACH COMBO'S
OWN FULL account (tsmom_exit_variants.py's own single-combo convention, for apples-to-
apples comparison with H_TP_EXIT/H_WIDE_TRAIL) -- NOT what's actually running on Propr
Beta. tsmom_beta_live.py splits ONE account into quarter-weighted shares of the 4
locked combos simultaneously (the structure H_JOINT_PORTFOLIO/H_DIRECTIONAL/
H_BTC_LEADER already modeled). User asked directly: what does the per-combo speed
change actually mean for the real pooled account? This answers that, reusing
tsmom_joint_portfolio_barrier_sim.py's pooled machinery completely unmodified.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_walkforward import walk_forward_one
from tsmom_joint_portfolio_barrier_sim import (
    LOCKED, WEIGHT, BLOCK_LEN_DAYS, N_SIMS, CYCLE_CAP_DAYS, SIM_YEARS,
    simulate_pooled_path, simulate_renewal_pooled, build_daily_series,
)
from tsmom_exit_variants import _real_history_series, ATR_MULT_BASE
from tsmom_partial_scaleout_variant import vol_target_returns_partial_scaleout

OUT = Path(__file__).parent / "output"

CANDIDATES = {
    "baseline (no scale-out)": None,
    "light (trig=1.0x, bank=25%)": {"atr_mult": ATR_MULT_BASE, "trigger_atr_mult": 1.0, "bank_frac": 0.25},
    "moderate (trig=1.5x, bank=50%)": {"atr_mult": ATR_MULT_BASE, "trigger_atr_mult": 1.5, "bank_frac": 0.50},
    "heavy (trig=1.0x, bank=75%)": {"atr_mult": ATR_MULT_BASE, "trigger_atr_mult": 1.0, "bank_frac": 0.75},
    "heavy-wide (trig=2.0x, bank=75%)": {"atr_mult": ATR_MULT_BASE, "trigger_atr_mult": 2.0, "bank_frac": 0.75},
}


def build_daily_series_partial(params: dict, wf_cache: dict) -> pd.DataFrame:
    """Parallel to tsmom_joint_portfolio_barrier_sim.build_daily_series, using the
    partial-scaleout signal instead of the plain one -- same OOS bar_range, same k/WEIGHT
    scaling, same daily resample/inner-join convention."""
    cols = {}
    for (coin, interval), (lb, k) in LOCKED.items():
        wf = wf_cache[(coin, interval)]
        s = _real_history_series(coin, interval, lb, vol_target_returns_partial_scaleout, **params)
        test = s.iloc[wf["bar_range"][0]:wf["bar_range"][1]]
        daily = test.resample("1D").sum() * k * WEIGHT
        cols[f"{coin}_{interval}"] = daily
    return pd.DataFrame(cols).dropna(how="any")


if __name__ == "__main__":
    wf_cache = {combo: walk_forward_one(*combo) for combo in LOCKED}

    rows = []
    for name, params in CANDIDATES.items():
        daily_df = build_daily_series() if params is None else build_daily_series_partial(params, wf_cache)
        daily_mat = daily_df.to_numpy()
        combined_hist = daily_df.sum(axis=1)
        sharpe = combined_hist.mean() / combined_hist.std() * np.sqrt(365) if combined_hist.std() > 0 else np.nan

        rng = np.random.default_rng(20260907)  # same seed as every other pooled comparison in this project
        outcomes = [simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng) for _ in range(N_SIMS)]
        outcome_names = [o[0] for o in outcomes]
        days_all = np.array([o[1] for o in outcomes if o[0] != "unresolved"])
        days_pass = np.array([o[1] for o in outcomes if o[0] == "pass"])

        row = {"candidate": name, "n_days_hist": len(daily_df), "sharpe": sharpe}
        for oc in ["pass", "fail_static", "fail_daily", "unresolved"]:
            row[oc] = outcome_names.count(oc) / N_SIMS
        row["median_days_all"] = float(np.median(days_all)) if len(days_all) else float("nan")
        row["median_days_pass"] = float(np.median(days_pass)) if len(days_pass) else float("nan")

        # $/month across repeated eval cycles -- the single number that combines
        # "how good" (pass rate, payout) and "how fast" (resolution speed) into the
        # metric the user actually cares about: rate of extracting money over time.
        rng2 = np.random.default_rng(20260908)  # same second-seed convention as tsmom_joint_portfolio_barrier_sim.py
        renewals = [simulate_renewal_pooled(daily_mat, BLOCK_LEN_DAYS, SIM_YEARS * 365, rng2) for _ in range(N_SIMS)]
        per_month = np.array([r["net_cash"] / (r["days"] / 30.44) for r in renewals])
        row["dollars_per_month_mean"] = per_month.mean()
        row["dollars_per_month_median"] = np.median(per_month)
        row["p_net_negative"] = np.mean(per_month < 0)
        row["cycles_per_yr"] = np.mean([( r["n_pass"] + r["n_fail"]) / (r["days"] / 365) for r in renewals])

        rows.append(row)
        print(f"  done: {name}")

    df = pd.DataFrame(rows).set_index("candidate")
    fmt = {c: "{:.1%}".format for c in ["pass", "fail_static", "fail_daily", "unresolved", "p_net_negative"]}
    fmt["sharpe"] = "{:.2f}".format
    fmt["median_days_all"] = "{:.0f}".format
    fmt["median_days_pass"] = "{:.0f}".format
    fmt["dollars_per_month_mean"] = "${:,.0f}".format
    fmt["dollars_per_month_median"] = "${:,.0f}".format
    fmt["cycles_per_yr"] = "{:.2f}".format
    print("\n=== Pooled $100K account (quarter-weighted, 4 locked combos together) ===")
    print(df[["sharpe", "pass", "fail_static", "median_days_all"]].to_string(formatters=fmt))
    print("\n=== $/month across repeated eval cycles (renewal sim, sweep-on-pass, same convention as H_JOINT_PORTFOLIO) ===")
    print(df[["dollars_per_month_mean", "dollars_per_month_median", "p_net_negative", "cycles_per_yr"]].to_string(formatters=fmt))

    OUT.mkdir(exist_ok=True)
    df.reset_index().to_json(OUT / "tsmom_partial_scaleout_pooled_account.json", orient="records", indent=2)
