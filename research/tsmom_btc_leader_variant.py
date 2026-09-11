"""
H_BTC_LEADER: BTC's momentum sign drives ETH/SOL's entry direction (session 27,
user-proposed from live Beta observation that BTC/ETH/SOL momentum moves together and
the resulting positions end up correlated regardless of direction -- so trading the
spread between them isn't practical, and leaning into the co-movement (BTC leads,
ETH/SOL follow) might do better than three independent signals that mostly agree
anyway.

This is NOT H_DIRECTIONAL (session 22/23, tsmom_directional_variants.py). That script
tested a different mechanism for the same underlying observation: restricting each
combo's OWN signal to trade only one direction (long-only/short-only), and found it
HURTS pooled P(pass) because cross-combo correlation rises (0.70-0.95 vs. 0.50-0.75 for
long-short) while diversification-driven pass probability collapses outside trending
regimes (81.0% -> 23.9%/46.2%). This idea doesn't restrict any combo's direction -- BTC
keeps trading its own independent long-short signal, unchanged. Only ETH's and SOL's
entry DIRECTION is replaced with BTC's causally-aligned momentum sign; their own ATR-
14/2.5x trailing stop, vol-targeted sizing, and locked leverage k stay exactly as today.
Empirically distinct question, same joint-portfolio machinery, and the same general risk
(forcing agreement across combos can raise correlation and hurt P(pass)) -- measured
here, not assumed.

Method: BTC's own per-bar direction series (`_strat_ret_series_with_dir`, a parallel
copy of tsmom_walkforward._strat_ret_series that also returns pos_dir) is causally
aligned onto ETH's and SOL's own (slower) bar timestamps via `align_leader_signal`
(pandas.merge_asof, direction="backward": each follower bar gets the most recent BTC
direction known AT OR BEFORE that bar's own open -- never a future BTC value).
`_strat_ret_series_btc_led` then runs ETH's/SOL's own entry+ATR-stop+vol-target loop
unchanged except that the entry condition checks the aligned BTC direction instead of
that coin's own momentum sign. The resulting per-combo OOS daily-return series (same
walk_forward_one bar_range discipline as every other p_pass number here) feed the exact
same joint-portfolio pooled-barrier machinery as tsmom_joint_portfolio_barrier_sim.py /
tsmom_directional_variants.py, so P(pass), correlation, and Sharpe are directly
comparable to those scripts' already-published numbers.

Caveats, stated plainly:
- BTC_4h's own candle history starts 2024-05-20, later than ETH_12h/SOL_12h's history.
  Verified at runtime (not assumed): zero bars in any LOCKED combo's own walk-forward
  OOS test slice fall before that date, so no follower bar in this comparison is ever
  flat-filled for lack of leader data. Followers DO still see the leader as flat
  ~4.5-6.8% of the time within their OOS slice -- verified to be BTC's own genuine flat
  periods (BTC itself sits flat, i.e. pos_dir==0, ~4.8-5.3% of the time), not a
  coverage gap or an alignment bug -- a follower simply stays flat too until BTC's
  signal (re-)establishes a direction.
- Because ETH/SOL still run their OWN ATR trailing stop independently of BTC, a combo
  can hold a stale direction for a while after BTC's own signal has already flipped
  (BTC's new direction only takes effect for a follower the next time that follower is
  flat and re-checks entry) -- deliberate (mechanics stay untouched per coin), and
  exactly what the netting/trade-count diagnostics below are built to surface.
- Same common-calendar-overlap caveat as H_JOINT_PORTFOLIO: BTC_4h's own OOS test slice
  is the binding constraint on the pooled sim's history length, unchanged by this
  variant.

Scope: research only. tsmom_beta_live.py, tsmom_paper_trader.py, and the locked
deployment config are not touched anywhere in this file -- LOCKED is only imported/read.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m
from tsmom_walkforward import walk_forward_one, _strat_ret_series
from tsmom_barrier_sim import TARGET, STATIC_FLOOR, DAILY_LOSS
from tsmom_joint_portfolio_barrier_sim import (
    LOCKED, WEIGHT, BLOCK_LEN_DAYS, N_SIMS, CYCLE_CAP_DAYS, simulate_pooled_path,
    build_daily_series,
)

OUT = Path(__file__).parent / "output"
LEADER = ("BTC", "4h")
EPISODE = ("2026-08-17", "2026-08-27")  # same rally-excision window as tsmom_gap_hypothesis.py etc.


def _strat_ret_series_with_dir(coin: str, interval: str, lookback_bars: int
                                 ) -> tuple[pd.Series, pd.Series]:
    """Parallel copy of tsmom_walkforward._strat_ret_series that also returns the
    per-bar position-direction series (that function only returns strat_ret). Used only
    to get BTC's own direction series as the leader signal -- BTC's return series itself
    is still taken from the original _strat_ret_series (imported, unmodified) everywhere
    else in this file, so this copy's own correctness is checked against it directly
    (see the sanity assertion in __main__) rather than trusted on its own."""
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
            if (not np.isnan(mom_a[t]) and mom_a[t] != 0 and not np.isnan(atr_a[t])
                    and not np.isnan(vol_a[t]) and vol_a[t] > 0):
                pos_dir = 1 if mom_a[t] > 0 else -1
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
    return (pd.Series(strat_ret, index=df.index).iloc[1:],
            pd.Series(pos_dir_arr, index=df.index).iloc[1:])


def align_leader_signal(leader_dir: pd.Series, follower_index: pd.DatetimeIndex) -> np.ndarray:
    """Causally aligns the leader's (BTC's) per-bar direction onto a follower's own
    (slower) bar timestamps: each follower bar gets the most recent leader direction
    known AT OR BEFORE that bar's own open -- never a future leader value. Both indices
    are candle-open timestamps (tsmom_hypothesis.load_candles), directly comparable, no
    timezone/offset issues. merge_asof(direction="backward") is used instead of index
    arithmetic (e.g. follower bar // 2) so it degrades gracefully (falls back to the
    prior known bar, never raises) if either series ever has a gap. Follower bars before
    the leader's own history starts get NaN -> filled with 0 (flat), never coerced to
    +-1."""
    left = pd.DataFrame({"t": follower_index})
    right = leader_dir.rename("leader_dir").reset_index().rename(columns={leader_dir.index.name or "index": "t"})
    merged = pd.merge_asof(left, right, on="t", direction="backward")
    return merged["leader_dir"].fillna(0.0).to_numpy()


def _strat_ret_series_btc_led(coin: str, interval: str, leader_dir_aligned: np.ndarray,
                                atr_k: int = 14, atr_mult: float = 2.5) -> pd.Series:
    """Parallel copy of the ATR-stop/vol-target loop for a FOLLOWER coin (ETH/SOL): same
    sizing/exit mechanics as _strat_ret_series, but the entry condition's own-momentum-
    sign check is replaced by the causally-aligned BTC direction. `leader_dir_aligned`
    must already be index-aligned to df.index (see align_leader_signal)."""
    cost = m.ROUND_TRIP_COST
    df = m.load_candles(coin, interval)
    log_ret = np.log1p(df["c"].pct_change())
    tr = np.maximum(df["h"] - df["l"], np.maximum((df["h"] - df["c"].shift(1)).abs(),
                                                    (df["l"] - df["c"].shift(1)).abs()))
    atr = tr.rolling(atr_k).mean().shift(1)
    realized_vol = log_ret.rolling(m.VOL_LOOKBACK_BARS).std().shift(1)
    bars_per_year = m.BARS_PER_YEAR[interval]
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(bars_per_year)

    close, high, low = df["c"].to_numpy(), df["h"].to_numpy(), df["l"].to_numpy()
    atr_a, vol_a = atr.to_numpy(), realized_vol.to_numpy()
    n = len(df)
    strat_ret = np.zeros(n)
    pos_size, pos_dir, stop_level = 0.0, 0, np.nan
    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            sig = int(leader_dir_aligned[t])
            if sig != 0 and not np.isnan(atr_a[t]) and not np.isnan(vol_a[t]) and vol_a[t] > 0:
                pos_dir = sig
                pos_size = min(target_vol_per_bar / vol_a[t], m.LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr_a[t]
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
                        stop_level = max(stop_level, close[t] - atr_mult * atr_a[t])
                    else:
                        stop_level = min(stop_level, close[t] + atr_mult * atr_a[t])
    return pd.Series(strat_ret, index=df.index).iloc[1:]


def build_daily_series_btc_leader(wf_cache: dict) -> pd.DataFrame:
    """BTC column: its own unchanged signal (_strat_ret_series, imported). ETH/SOL
    columns: BTC-led direction (_strat_ret_series_btc_led). Each sliced to its own
    walk-forward OOS bar_range, scaled by LOCKED k and WEIGHT, resampled daily,
    inner-joined -- same convention as build_daily_series / build_daily_series_variant."""
    _, leader_dir_full = _strat_ret_series_with_dir(*LEADER, lookback_bars=1)
    cols = {}
    for (coin, interval), (lb, k) in LOCKED.items():
        wf = wf_cache[(coin, interval)]
        if (coin, interval) == LEADER:
            s = _strat_ret_series(coin, interval, lb)
        else:
            df = m.load_candles(coin, interval)
            aligned = align_leader_signal(leader_dir_full, df.index)[1:]  # drop row 0, matches df.index[1:]
            s = _strat_ret_series_btc_led(coin, interval, aligned)
        test = s.iloc[wf["bar_range"][0]:wf["bar_range"][1]]
        daily = test.resample("1D").sum() * k * WEIGHT
        cols[f"{coin}_{interval}"] = daily
    return pd.DataFrame(cols).dropna(how="any")


def netting_diagnostic_variant(name: str, pos_dir_by_combo: dict[tuple, pd.Series]) -> None:
    """Full-history (not just OOS slice) netting diagnostic, adapted from
    tsmom_directional_variants.netting_diagnostic: per day, how much of the 4 combos'
    gross weighted exposure survives after summing signed exposure."""
    exposure_cols = {}
    for (coin, interval), pos_dir in pos_dir_by_combo.items():
        lb, k = LOCKED[(coin, interval)]
        daily_dir = pos_dir.resample("1D").last()
        exposure_cols[f"{coin}_{interval}"] = daily_dir * k * WEIGHT
    exp_df = pd.DataFrame(exposure_cols).dropna(how="any")
    gross = exp_df.abs().sum(axis=1)
    net = exp_df.sum(axis=1).abs()
    survival = net / gross.replace(0, np.nan)
    active_days = gross > 0
    print(f"\n=== Netting diagnostic [{name}] (full history, "
          f"{active_days.sum()}/{len(exp_df)} days with >=1 combo active) ===")
    print(f"  mean fraction of gross exposure surviving netting: "
          f"{survival[active_days].mean():.1%}")
    print(f"  fully-flat days (net==0 while gross>0): "
          f"{((gross > 0) & (net < 1e-9)).sum()} / {active_days.sum()} active days "
          f"({((gross > 0) & (net < 1e-9)).sum() / active_days.sum():.1%})")
    print(f"  all-agree days (net==gross): "
          f"{(np.isclose(net, gross) & active_days).sum()} / {active_days.sum()} active days "
          f"({(np.isclose(net, gross) & active_days).sum() / active_days.sum():.1%})")


def run_variant(name: str, daily_df: pd.DataFrame, rng: np.random.Generator,
                 ex_rally: bool = False) -> dict:
    if ex_rally:
        mask = ~((daily_df.index >= EPISODE[0]) & (daily_df.index <= EPISODE[1]))
        daily_df = daily_df[mask]
    corr = daily_df.corr()
    combined_hist = daily_df.sum(axis=1)
    sharpe = combined_hist.mean() / combined_hist.std() * np.sqrt(365) if combined_hist.std() > 0 else np.nan

    daily_mat = daily_df.to_numpy()
    outcomes = [simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng) for _ in range(N_SIMS)]
    outcome_names = [o[0] for o in outcomes]
    days_to_res = np.array([o[1] for o in outcomes if o[0] != "unresolved"])

    label = f"{name} ({'ex-rally' if ex_rally else 'full'})"
    print(f"\n=== {label} (n_days={len(daily_df)}, {daily_df.index.min().date()} -> "
          f"{daily_df.index.max().date()}) ===")
    print("Pairwise correlation of quarter-weighted daily strategy returns:")
    print(corr.round(2).to_string())
    print(f"Pooled daily return: mean={combined_hist.mean():.5f}, std={combined_hist.std():.5f}, "
          f"ann. Sharpe={sharpe:.2f}")
    res = {"variant": name, "sample": "ex_rally" if ex_rally else "full"}
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

    # --- Sanity check: the parallel-copy BTC series must reproduce the imported,
    # already-validated one bit-for-bit before anything downstream can be trusted. ---
    btc_lb = LOCKED[LEADER][0]
    ref_ret = _strat_ret_series(*LEADER, btc_lb)
    check_ret, leader_dir_full = _strat_ret_series_with_dir(*LEADER, btc_lb)
    assert np.allclose(ref_ret.to_numpy(), check_ret.to_numpy()), \
        "BTC parallel-copy return series does not match _strat_ret_series -- fix before trusting variant numbers"
    print(f"Sanity check OK: _strat_ret_series_with_dir reproduces BTC {LEADER[1]} "
          f"_strat_ret_series bit-for-bit ({len(ref_ret)} bars).")

    # --- No-lookahead / coverage checks on the leader-signal alignment ---
    for (coin, interval), (lb, k) in LOCKED.items():
        if (coin, interval) == LEADER:
            continue
        df = m.load_candles(coin, interval)
        aligned_full = align_leader_signal(leader_dir_full, df.index)
        wf = wf_cache[(coin, interval)]
        aligned_oos = aligned_full[1:][wf["bar_range"][0]:wf["bar_range"][1]]
        flat_frac = (aligned_oos == 0).mean()
        pre_leader = (df.index[1:][wf["bar_range"][0]:wf["bar_range"][1]] < leader_dir_full.index[0]).sum()
        print(f"  {coin}_{interval}: leader-signal flat fraction within its own OOS test "
              f"slice = {flat_frac:.2%} (bars before BTC leader history even starts: "
              f"{pre_leader}, expect 0)")

    sample_idx = m.load_candles("ETH", "12h").index
    sample_aligned = pd.Series(align_leader_signal(leader_dir_full, sample_idx), index=sample_idx)
    print("\nSample ETH_12h <- BTC_4h alignment rows inside the Aug 17-27 rally window:")
    print(sample_aligned.loc[EPISODE[0]:EPISODE[1]].head(8).to_string())

    # --- Baseline ("today") vs. BTC-leader variant ---
    baseline_df = build_daily_series()
    variant_df = build_daily_series_btc_leader(wf_cache)

    baseline_pos_dir = {}
    variant_pos_dir = {}
    for (coin, interval), (lb, k) in LOCKED.items():
        if (coin, interval) == LEADER:
            _, pd_base = _strat_ret_series_with_dir(coin, interval, lb)
            baseline_pos_dir[(coin, interval)] = pd_base
            variant_pos_dir[(coin, interval)] = pd_base
        else:
            _, pd_base = _strat_ret_series_with_dir(coin, interval, lb)
            baseline_pos_dir[(coin, interval)] = pd_base
            df = m.load_candles(coin, interval)
            aligned = align_leader_signal(leader_dir_full, df.index)[1:]
            variant_pos_dir[(coin, interval)] = pd.Series(aligned, index=df.index[1:])

    netting_diagnostic_variant("baseline (today)", baseline_pos_dir)
    netting_diagnostic_variant("btc_leader", variant_pos_dir)

    rows = []
    for name, daily_df in [("baseline", baseline_df), ("btc_leader", variant_df)]:
        for ex_rally in [False, True]:
            rng = np.random.default_rng(20260907)  # fresh per (variant, sample) -> comparable draws
            rows.append(run_variant(name, daily_df, rng, ex_rally=ex_rally))

    # --- Trade-count sanity check (holding-period magnitude) ---
    print("\n=== Direction-change (trade) counts, full history ===")
    for combo in LOCKED:
        n_base = int((baseline_pos_dir[combo].diff().fillna(0) != 0).sum())
        n_var = int((variant_pos_dir[combo].diff().fillna(0) != 0).sum())
        print(f"  {combo[0]}_{combo[1]}: baseline={n_base}, btc_leader={n_var}")

    print("\n=== Summary ===")
    df = pd.DataFrame(rows).set_index(["variant", "sample"])
    fmt = {c: "{:.1%}".format for c in ["pass", "fail_static", "fail_daily", "unresolved"]}
    fmt["sharpe"] = "{:.2f}".format
    fmt["median_days"] = "{:.0f}".format
    print(df.to_string(formatters=fmt))

    OUT.mkdir(exist_ok=True)
    df.reset_index().to_json(OUT / "tsmom_btc_leader_variant.json", orient="records", indent=2)
    print(f"\nSaved to {OUT / 'tsmom_btc_leader_variant.json'}")
