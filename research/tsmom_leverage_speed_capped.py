"""
H_LEVERAGE_SPEED, per-bar cap verification (session 28 continued). The prior sweep
(tsmom_leverage_speed_sweep.py) applied k_extra as a linear rescale of the ALREADY-
REALIZED OOS daily-return series -- it never re-checked whether LOCKED_k * k_extra,
applied to the per-bar vol-targeted size BEFORE that size's own cap, would have been
clipped at the raw per-asset LEVERAGE_CAP (BTC/ETH 5x, SOL 2x) on any actual bar. This
re-runs the signal properly: `_strat_ret_series_leveraged` is a parallel copy of
tsmom_walkforward._strat_ret_series with position sizing capped exactly the way
tsmom_barrier_sim.simulate_path's own `_size` helper already does for a leverage
overlay -- min(raw vol-target size, CAP), times LOCKED_k * k_extra, capped at CAP AGAIN:

    pos_size = min(min(target_vol_per_bar / realized_vol, CAP) * locked_k * k_extra, CAP)

k_extra=1.0 must reproduce _strat_ret_series (i.e. today's production signal) bar-for-
bar, since LOCKED_k alone never binds the cap today (checked below). Narrowed to
k_extra in {1.5, 2.0, 2.5, 3.0} per the user's own follow-up focus on that region
(H_LEVERAGE_SPEED's "pass within a short fixed window" sweet spot was 2-3x).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m
from tsmom_walkforward import walk_forward_one, _strat_ret_series
from tsmom_joint_portfolio_barrier_sim import (
    LOCKED, WEIGHT, BLOCK_LEN_DAYS, N_SIMS, CYCLE_CAP_DAYS, SIM_YEARS,
    simulate_pooled_path, simulate_renewal_pooled, build_daily_series,
)

OUT = Path(__file__).parent / "output"
K_EXTRA_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
HORIZON_GRID_DAYS = [7, 10, 14, 21, 30, 60, 90]


def _strat_ret_series_leveraged(coin: str, interval: str, lookback_bars: int, locked_k: float,
                                  k_extra: float, round_trip_cost: float | None = None
                                  ) -> tuple[pd.Series, float]:
    """Parallel copy of tsmom_walkforward._strat_ret_series with position sizing capped
    exactly like tsmom_barrier_sim.simulate_path's `_size` overlay convention:
    pos_size = min(min(raw_vol_target_size, CAP) * locked_k * k_extra, CAP). Returns
    (strat_ret series, clip_frac) where clip_frac is the fraction of ACTIVE (nonzero
    vol-target-eligible) bars where the final cap actually bound -- i.e. where the naive
    linear-rescale approximation (tsmom_leverage_speed_sweep.py) would have overstated
    the true position size."""
    cost = m.ROUND_TRIP_COST if round_trip_cost is None else round_trip_cost
    df = m.load_candles(coin, interval)
    log_ret = np.log1p(df["c"].pct_change())
    mom = df["c"].pct_change(lookback_bars).shift(1)
    tr = np.maximum(df["h"] - df["l"], np.maximum((df["h"] - df["c"].shift(1)).abs(),
                                                    (df["l"] - df["c"].shift(1)).abs()))
    atr = tr.rolling(14).mean().shift(1)
    realized_vol = log_ret.rolling(m.VOL_LOOKBACK_BARS).std().shift(1)
    bars_per_year = m.BARS_PER_YEAR[interval]
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(bars_per_year)
    cap = m.LEVERAGE_CAP[coin]

    close, high, low = df["c"].to_numpy(), df["h"].to_numpy(), df["l"].to_numpy()
    mom_a, atr_a, vol_a = mom.to_numpy(), atr.to_numpy(), realized_vol.to_numpy()
    n = len(df)
    strat_ret = np.zeros(n)
    pos_size, pos_dir, stop_level = 0.0, 0, np.nan
    n_active, n_clipped = 0, 0

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if (not np.isnan(mom_a[t]) and mom_a[t] != 0 and not np.isnan(atr_a[t])
                    and not np.isnan(vol_a[t]) and vol_a[t] > 0):
                pos_dir = 1 if mom_a[t] > 0 else -1
                raw_vol_size = target_vol_per_bar / vol_a[t]
                capped_vol_size = min(raw_vol_size, cap)
                scaled = capped_vol_size * locked_k * k_extra
                pos_size = min(scaled, cap)
                n_active += 1
                n_clipped += int(scaled > cap + 1e-12)
                stop_level = prev_close - pos_dir * 2.5 * atr_a[t]
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * cost / 2
        else:
            breached = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            if breached:
                strat_ret[t] = pos_dir * pos_size * np.log(stop_level / prev_close) - pos_size * cost / 2
                pos_dir, pos_size, stop_level = 0, 0.0, np.nan
            else:
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr_a[t]):
                    if pos_dir == 1:
                        stop_level = max(stop_level, close[t] - 2.5 * atr_a[t])
                    else:
                        stop_level = min(stop_level, close[t] + 2.5 * atr_a[t])
    clip_frac = n_clipped / n_active if n_active else 0.0
    return pd.Series(strat_ret, index=df.index).iloc[1:], clip_frac


def build_daily_series_leveraged(k_extra: float, wf_cache: dict) -> tuple[pd.DataFrame, dict]:
    cols = {}
    clip_fracs = {}
    for (coin, interval), (lb, locked_k) in LOCKED.items():
        wf = wf_cache[(coin, interval)]
        s, clip_frac = _strat_ret_series_leveraged(coin, interval, lb, locked_k, k_extra)
        clip_fracs[f"{coin}_{interval}"] = clip_frac
        test = s.iloc[wf["bar_range"][0]:wf["bar_range"][1]]
        daily = test.resample("1D").sum() * WEIGHT  # locked_k * k_extra already baked into pos_size above
        cols[f"{coin}_{interval}"] = daily
    return pd.DataFrame(cols).dropna(how="any"), clip_fracs


def resolution_within(days_to_res, outcome_names, horizons) -> dict:
    out = {}
    days_arr = np.array(days_to_res)
    outcomes_arr = np.array(outcome_names)
    for T in horizons:
        resolved = (outcomes_arr != "unresolved") & (days_arr <= T)
        out[f"p_resolved_le_{T}d"] = resolved.mean()
        out[f"p_pass_le_{T}d"] = ((outcomes_arr == "pass") & (days_arr <= T)).mean()
    return out


if __name__ == "__main__":
    wf_cache = {combo: walk_forward_one(*combo) for combo in LOCKED}

    print("=== Sanity check: k_extra=1.0 reproduces build_daily_series() (today's production baseline) ===")
    # NOTE: _strat_ret_series (imported, unmodified) is the RAW k=1 signal with no
    # LOCKED_k scaling at all -- not the right comparison target for k_extra=1.0, since
    # this function bakes locked_k in. The correct baseline is build_daily_series(),
    # which applies locked_k*WEIGHT post-hoc to the same raw series -- mathematically
    # equivalent to scaling pos_size by locked_k inside the loop UNLESS the raw cap
    # binds differently, which is exactly what this whole check is verifying.
    ref_df = build_daily_series()
    test_df, clip_fracs_1 = build_daily_series_leveraged(1.0, wf_cache)
    for col in ref_df.columns:
        match = np.allclose(ref_df[col].to_numpy(), test_df[col].to_numpy(), atol=1e-10)
        print(f"  {col}: {'OK' if match else 'MISMATCH'} (clip_frac at k_extra=1.0: {clip_fracs_1[col]:.2%})")
        if not match:
            sys.exit("k_extra=1.0 baseline-reproduction sanity check failed.")

    rows = []
    for k_extra in K_EXTRA_GRID:
        daily_df, clip_fracs = build_daily_series_leveraged(k_extra, wf_cache)
        daily_mat = daily_df.to_numpy()
        combined_hist = daily_df.sum(axis=1)
        sharpe = combined_hist.mean() / combined_hist.std() * np.sqrt(365) if combined_hist.std() > 0 else np.nan

        rng = np.random.default_rng(20260907)
        outcomes = [simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng) for _ in range(N_SIMS)]
        outcome_names = [o[0] for o in outcomes]
        days_to_res = [o[1] for o in outcomes]

        row = {"k_extra": k_extra, "sharpe": sharpe, "max_clip_frac": max(clip_fracs.values())}
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
        print(f"  done: k_extra={k_extra}  per-combo clip_frac={ {k: f'{v:.1%}' for k, v in clip_fracs.items()} }")

    df = pd.DataFrame(rows).set_index("k_extra")
    fmt = {c: "{:.1%}".format for c in ["pass", "fail_static", "fail_daily", "unresolved", "max_clip_frac"]}
    fmt.update({c: "{:.1%}".format for c in df.columns if c.startswith("p_resolved") or c.startswith("p_pass_le")})
    fmt["sharpe"] = "{:.2f}".format
    fmt["median_days_all"] = "{:.0f}".format
    fmt["dollars_per_month_mean"] = "${:,.0f}".format
    fmt["cycles_per_yr"] = "{:.2f}".format

    print("\n=== Properly-capped leverage sweep, pooled $100K account ===")
    print(df[["sharpe", "pass", "fail_static", "median_days_all", "cycles_per_yr",
               "dollars_per_month_mean", "max_clip_frac"]].to_string(formatters=fmt))

    print("\n=== Resolved/pass-within-T-days ===")
    cols = [f"p_resolved_le_{T}d" for T in HORIZON_GRID_DAYS] + [f"p_pass_le_{T}d" for T in HORIZON_GRID_DAYS]
    print(df[cols].to_string(formatters=fmt))

    OUT.mkdir(exist_ok=True)
    df.reset_index().to_json(OUT / "tsmom_leverage_speed_capped.json", orient="records", indent=2)
