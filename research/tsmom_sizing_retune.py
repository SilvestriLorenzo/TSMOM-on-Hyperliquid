"""
H_TSMOM sizing-overlay retune (session 11), continuing tsmom_walkforward.py. Session 10
tested the user's proposed sizing overlay (Magdon-Ismail & Atiya expected-max-drawdown
calibration, Grossman-Zhou-style cushion-proportional de-risking) only at its most
literal settings, and both made things worse than plain fixed-vol sizing. HYPOTHESES.md's
own conclusion: the productive next step is tuning these away from that literal framing,
not abandoning the signal. This does that tuning, evaluated against tsmom_walkforward's
walk-forward-honest out-of-sample baseline (train-selected lookback, test-period-only
bootstrap source/leverage) rather than the original in-sample-fit baseline, so any
"improvement" found here isn't re-inflated by the same selection bias the walk-forward
work exists to fix.

Two knobs, swept independently against the honest baseline:

1. MI&A target_mdd x calibration horizon. calibrate_leverage_for_target_mdd needs no
   root-finding (E[MDD] scales linearly with leverage) -- session 10 only ever tried
   target_mdd=1.5% at horizon_days up to 1095, both far more conservative than the
   already-validated fixed-vol leverage's own natural drawdown profile supports. Sweep
   wider on both.

2. Budget-normalized CPPI de-risking (tsmom_barrier_sim.simulate_path's new
   cppi_budget_c parameter). Session 10's literal overlay normalized the cushion
   against current EQUITY (~1.0), not against the actual 3%-of-equity drawdown BUDGET
   it was meant to protect -- an implicit ~33x-too-conservative ratio baked into the
   formula itself, independent of any multiplier, which is why it drove every path to
   "unresolved." The corrected version here normalizes against the real budget and
   sweeps how close to the floor de-risking should start engaging.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_walkforward import walk_forward_one, COMBOS
from tsmom_barrier_sim import run_barrier_sim

OUT = Path(__file__).parent / "output"
MIA_TARGET_MDD = [0.015, 0.0225, 0.03, 0.045]
MIA_HORIZON_DAYS = [30, 90, 180, 365]
CPPI_C = [0.1, 0.25, 0.5, 1.0]
N_PATHS = 400


def sweep_mia(coin: str, interval: str, lb_star: int, bar_range: tuple[int, int]) -> list[dict]:
    rows = []
    for target_mdd in MIA_TARGET_MDD:
        for horizon_days in MIA_HORIZON_DAYS:
            sim = run_barrier_sim(coin, interval, lb_star, n_paths=N_PATHS, cppi=False,
                                   leverage=None, bar_range=bar_range,
                                   mia_target_mdd=target_mdd, mia_horizon_days=horizon_days)
            rows.append({"coin": coin, "interval": interval, "target_mdd": target_mdd,
                         "horizon_days": horizon_days, "leverage": sim["leverage"],
                         "p_pass": sim["pass"], "fail_static": sim["fail_static"],
                         "fail_daily": sim["fail_daily"], "unresolved": sim["unresolved"]})
    return rows


def sweep_cppi(coin: str, interval: str, lb_star: int, bar_range: tuple[int, int],
               fixed_vol_leverage: float) -> list[dict]:
    rows = []
    for c in CPPI_C:
        sim = run_barrier_sim(coin, interval, lb_star, n_paths=N_PATHS, cppi=False,
                               leverage=fixed_vol_leverage, bar_range=bar_range,
                               cppi_budget_c=c)
        rows.append({"coin": coin, "interval": interval, "cppi_budget_c": c,
                     "leverage": fixed_vol_leverage, "p_pass": sim["pass"],
                     "fail_static": sim["fail_static"], "fail_daily": sim["fail_daily"],
                     "unresolved": sim["unresolved"]})
    return rows


if __name__ == "__main__":
    # Reruns tsmom_walkforward's honest result per combo -- gives the train-selected
    # lookback, test bar_range, and fixed-vol leverage/p_pass every sweep is compared to.
    baselines = {(coin, interval): walk_forward_one(coin, interval) for coin, interval in COMBOS}

    mia_rows, cppi_rows = [], []
    for (coin, interval), wf in baselines.items():
        bar_range = wf["bar_range"]
        mia_rows += sweep_mia(coin, interval, wf["lb_star"], bar_range)
        cppi_rows += sweep_cppi(coin, interval, wf["lb_star"], bar_range, wf["oos_leverage_k"])

    mia_df = pd.DataFrame(mia_rows)
    cppi_df = pd.DataFrame(cppi_rows)
    OUT.mkdir(exist_ok=True)
    mia_df.to_json(OUT / "tsmom_mia_sweep.json", orient="records", indent=2)
    cppi_df.to_json(OUT / "tsmom_cppi_sweep.json", orient="records", indent=2)

    fmt = {"leverage": "{:.3f}".format, "p_pass": "{:.1%}".format,
           "fail_static": "{:.1%}".format, "fail_daily": "{:.1%}".format,
           "unresolved": "{:.1%}".format, "target_mdd": "{:.2%}".format,
           "cppi_budget_c": "{:.2f}".format}

    print("=== MI&A target_mdd x horizon sweep ===")
    print(mia_df.to_string(index=False, formatters=fmt))

    print("\n=== Budget-normalized CPPI de-risking sweep ===")
    print(cppi_df.to_string(index=False, formatters=fmt))

    print("\n=== Honest (walk-forward) baseline p_pass, no sizing overlay ===")
    for (coin, interval), wf in baselines.items():
        print(f"  {coin} {interval}: lb*={wf['lb_star']}, leverage={wf['oos_leverage_k']:.3f}, "
              f"p_pass={wf['oos_p_pass']:.1%}")

    best_mia = mia_df.loc[mia_df.groupby(["coin", "interval"])["p_pass"].idxmax()]
    best_cppi = cppi_df.loc[cppi_df.groupby(["coin", "interval"])["p_pass"].idxmax()]
    print("\n=== Best MI&A config per combo vs. honest baseline ===")
    for _, row in best_mia.iterrows():
        base = baselines[(row["coin"], row["interval"])]["oos_p_pass"]
        verdict = "BEATS baseline" if row["p_pass"] > base else "does not beat baseline"
        print(f"  {row['coin']} {row['interval']}: target_mdd={row['target_mdd']:.2%}, "
              f"horizon={row['horizon_days']:.0f}d -> p_pass={row['p_pass']:.1%} "
              f"(baseline {base:.1%}) -- {verdict}")

    print("\n=== Best budget-CPPI config per combo vs. honest baseline ===")
    for _, row in best_cppi.iterrows():
        base = baselines[(row["coin"], row["interval"])]["oos_p_pass"]
        verdict = "BEATS baseline" if row["p_pass"] > base else "does not beat baseline"
        print(f"  {row['coin']} {row['interval']}: c={row['cppi_budget_c']:.2f} -> "
              f"p_pass={row['p_pass']:.1%} (baseline {base:.1%}) -- {verdict}")
