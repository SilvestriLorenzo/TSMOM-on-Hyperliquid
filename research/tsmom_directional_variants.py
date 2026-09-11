"""
H_TSMOM long-only / short-only vs long-short comparison (session 22, user-prompted from
Beta live observation: "BTC short / ETH+SOL long often cancel each other and the
[pooled] strategy remains flat", since the three assets tend to co-move -- question is
whether letting the signal trade BOTH directions is actually costing pass-probability
relative to a directional-only variant, net of what's likely just short-sample noise in
the first few days of Beta live trading.

Method note -- this is NOT a post-hoc mask of the existing (both-directions) return
series. `_strat_ret_series_dirs` below duplicates tsmom_walkforward._strat_ret_series
(same discipline as tsmom_beta_live.py's signal_state: the validated function is left
untouched, this is a parallel copy) but adds an `allowed_dirs` filter applied AT ENTRY:
when flat and momentum's sign isn't in `allowed_dirs`, the position simply stays flat and
RE-CHECKS every subsequent bar, rather than being suppressed mid-trade. That distinction
matters concretely: in the real (both-directions) strategy, once a short is entered it is
LOCKED until its own ATR stop breaches, even if momentum flips positive mid-trade -- a
true long-only variant is free to enter the moment momentum turns positive instead of
waiting out the masked short's full duration. Masking the combined series post-hoc would
therefore understate how often long-only/short-only actually gets to trade.

Runs three variants (long_short = current production signal, long_only, short_only)
through the exact same joint-portfolio machinery as
tsmom_joint_portfolio_barrier_sim.py (same OOS test-slice discipline via
walk_forward_one, same LOCKED lb/k table, same quarter-weighting, same
correlation-preserving joint block bootstrap) to see whether restricting to one
direction changes (a) cross-combo correlation of daily returns -- the user's stated
netting hypothesis, (b) pooled Sharpe, (c) joint P(pass) on the pooled $100K account.

Also reports a direct, full-history "netting" diagnostic: for the current
both-directions signal, on each day how much of the 4 combos' gross (unsigned, weighted)
exposure survives after summing signed exposure -- i.e. exactly what the user watched
happen live, quantified over the whole backtest instead of a few days.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m
from tsmom_walkforward import walk_forward_one
from tsmom_barrier_sim import TARGET, STATIC_FLOOR, DAILY_LOSS
from tsmom_joint_portfolio_barrier_sim import (
    LOCKED, WEIGHT, BLOCK_LEN_DAYS, N_SIMS, CYCLE_CAP_DAYS, simulate_pooled_path,
)

VARIANTS = {"long_short": {1, -1}, "long_only": {1}, "short_only": {-1}}


def _strat_ret_series_dirs(coin: str, interval: str, lookback_bars: int,
                            allowed_dirs: set[int]) -> tuple[pd.Series, pd.Series]:
    """Returns (strat_ret, pos_dir) series, index-aligned to df.index[1:]. See module
    docstring: entry is filtered by allowed_dirs, not the resulting return series."""
    cost = m.ROUND_TRIP_COST
    df = m.load_candles(coin, interval)
    log_ret = np.log1p(df["c"].pct_change())
    mom = df["c"].pct_change(lookback_bars).shift(1)
    tr = np.maximum(df["h"] - df["l"], np.maximum((df["h"] - df["c"].shift(1)).abs(),
                                                    (df["l"] - df["c"].shift(1)).abs()))
    atr = tr.rolling(14).mean().shift(1)
    realized_vol = log_ret.rolling(m.VOL_LOOKBACK_BARS).std().shift(1)
    bars_per_year = m.BARS_PER_YEAR[interval]
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(bars_per_year)

    close, high, low = df["c"].to_numpy(), df["h"].to_numpy(), df["l"].to_numpy()
    mom_a, atr_a, vol_a = mom.to_numpy(), atr.to_numpy(), realized_vol.to_numpy()
    n = len(df)
    strat_ret = np.zeros(n)
    pos_dir_arr = np.zeros(n)
    pos_size, pos_dir, stop_level = 0.0, 0, np.nan
    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            sig = 0
            if not np.isnan(mom_a[t]) and mom_a[t] != 0:
                sig = 1 if mom_a[t] > 0 else -1
            if (sig in allowed_dirs and not np.isnan(atr_a[t])
                    and not np.isnan(vol_a[t]) and vol_a[t] > 0):
                pos_dir = sig
                pos_size = min(target_vol_per_bar / vol_a[t], m.LEVERAGE_CAP[coin])
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
        pos_dir_arr[t] = pos_dir
    idx = df.index[1:]
    return pd.Series(strat_ret, index=df.index).iloc[1:], pd.Series(pos_dir_arr, index=df.index).iloc[1:]


def build_daily_series_variant(allowed_dirs: set[int], wf_cache: dict) -> pd.DataFrame:
    cols = {}
    for (coin, interval), (lb, k) in LOCKED.items():
        wf = wf_cache[(coin, interval)]
        s, _ = _strat_ret_series_dirs(coin, interval, lb, allowed_dirs)
        test = s.iloc[wf["bar_range"][0]:wf["bar_range"][1]]
        daily = test.resample("1D").sum() * k * WEIGHT
        cols[f"{coin}_{interval}"] = daily
    return pd.DataFrame(cols).dropna(how="any")


def netting_diagnostic(wf_cache: dict) -> None:
    """Full-history (not just OOS slice), both-direction signal: per day, how much of
    the 4 combos' gross weighted exposure survives netting -- the user's live
    observation, quantified over the whole backtest."""
    exposure_cols = {}
    for (coin, interval), (lb, k) in LOCKED.items():
        _, pos_dir = _strat_ret_series_dirs(coin, interval, lb, {1, -1})
        daily_dir = pos_dir.resample("1D").last()  # end-of-day direction, one row/day
        exposure_cols[f"{coin}_{interval}"] = daily_dir * k * WEIGHT
    exp_df = pd.DataFrame(exposure_cols).dropna(how="any")
    gross = exp_df.abs().sum(axis=1)
    net = exp_df.sum(axis=1).abs()
    survival = net / gross.replace(0, np.nan)
    active_days = gross > 0
    print(f"\n=== Netting diagnostic (full history, both-directions signal, "
          f"{active_days.sum()}/{len(exp_df)} days with >=1 combo active) ===")
    print(f"  mean fraction of gross exposure surviving netting: "
          f"{survival[active_days].mean():.1%}")
    print(f"  fully-flat days (net==0 while gross>0, i.e. a full cancel): "
          f"{((gross > 0) & (net < 1e-9)).sum()} / {active_days.sum()} active days "
          f"({((gross > 0) & (net < 1e-9)).sum() / active_days.sum():.1%})")
    print(f"  all-agree days (net==gross, no cancellation at all): "
          f"{(np.isclose(net, gross) & active_days).sum()} / {active_days.sum()} active days "
          f"({(np.isclose(net, gross) & active_days).sum() / active_days.sum():.1%})")


def run_variant(name: str, allowed_dirs: set[int], wf_cache: dict, rng: np.random.Generator) -> dict:
    daily_df = build_daily_series_variant(allowed_dirs, wf_cache)
    corr = daily_df.corr()
    combined_hist = daily_df.sum(axis=1)
    sharpe = combined_hist.mean() / combined_hist.std() * np.sqrt(365) if combined_hist.std() > 0 else np.nan

    daily_mat = daily_df.to_numpy()
    outcomes = [simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng) for _ in range(N_SIMS)]
    outcome_names = [o[0] for o in outcomes]
    days_to_res = np.array([o[1] for o in outcomes if o[0] != "unresolved"])

    print(f"\n=== {name} (n_days={len(daily_df)}, {daily_df.index.min().date()} -> "
          f"{daily_df.index.max().date()}) ===")
    print("Pairwise correlation of quarter-weighted daily strategy returns:")
    print(corr.round(2).to_string())
    print(f"Pooled daily return: mean={combined_hist.mean():.5f}, std={combined_hist.std():.5f}, "
          f"ann. Sharpe={sharpe:.2f}")
    res = {"variant": name}
    for oc in ["pass", "fail_static", "fail_daily", "unresolved"]:
        frac = outcome_names.count(oc) / N_SIMS
        res[oc] = frac
        print(f"  {oc:>12}: {frac:.1%}")
    res["sharpe"] = sharpe
    res["median_days"] = float(np.median(days_to_res)) if len(days_to_res) else float("nan")
    print(f"  median days to resolution: {res['median_days']:.0f}")
    return res


if __name__ == "__main__":
    wf_cache = {combo: walk_forward_one(*combo) for combo in LOCKED}

    netting_diagnostic(wf_cache)

    rows = []
    for name, dirs in VARIANTS.items():
        rng = np.random.default_rng(20260907)  # same seed across variants -> comparable draws
        rows.append(run_variant(name, dirs, wf_cache, rng))

    print("\n=== Summary ===")
    df = pd.DataFrame(rows).set_index("variant")
    fmt = {c: "{:.1%}".format for c in ["pass", "fail_static", "fail_daily", "unresolved"]}
    fmt["sharpe"] = "{:.2f}".format
    fmt["median_days"] = "{:.0f}".format
    print(df.to_string(formatters=fmt))
