"""
H_POST_TP_REENTRY: the post-TP second-decision mechanism (session 25) -- the actual novel
research question this thread was built for. User's framing: the current strategy often
captures a real intra-window move (e.g. +5%) only to give it back before the next primary
bar-close decision (H_GIVEBACK, session 23, confirmed this is real and broad: median
give-back ratio 119% pooled). Idea: keep the existing entry signal exactly as-is, add a
take-profit as a FIRST-STAGE exit; once it fires, compute a fresh momentum reading and
take a SECOND, independent decision -- re-enter same direction (momentum holds), flip
(reversal), or go flat until the next primary bar-close signal -- instead of always going
flat the way a plain TP would.

Design note (resolved via a clarifying question before writing this): the raw window
from original entry to TP-fire is favorable BY CONSTRUCTION (that's what triggered the
TP) -- a score defined as "return over that whole window" is always positive in the
trade's own direction, so a literal reading would make the "flip to reversal" branch
unreachable. Resolved by computing the second-stage score as a FRESH short-lookback
momentum reading (same style as the base entry signal: sign/magnitude of price change
over a short trailing window), evaluated AT the TP-fire bar instead of over the whole
entry-to-TP window -- this can genuinely point either direction, since price may have
already begun reversing right as the TP tagged even though the overall trade was
profitable.

TP threshold is NOT re-searched here -- per the project's own sequencing (session 24,
H_TP_EXIT), it is fixed per combo to the already train-selected `alt`-mode value (BTC 4h
3.5xATR, ETH 8h 2.0xATR, ETH 12h 3.5xATR, SOL 12h 2.5xATR), reused here purely as a
STAGE-1 TRIGGER, not as a claim that TP is a good terminal exit on its own (session 24's
verdict was that it mostly isn't). What IS searched (train-only, per combo): score type
(raw log-return vs. vol-normalized/Sharpe-like), the short lookback window the score is
computed over, and the flat-band threshold eps separating hold/flip/flat.

Mechanics (deliberately NOT chained/recursive, to keep this a single well-defined
extra decision rather than open-ended complexity): a "base" trade (opened by the
ordinary primary signal) watches BOTH the existing ATR stop and the fixed TP -- if the
ATR stop fires first, the trade closes exactly as production does today, no second
decision. If the TP fires first, the trade closes at the TP level, a fresh short-lookback
score is computed using data through that same bar (already fully realized, no look-ahead
into future bars), and the decision is applied starting the NEXT bar: hold (re-enter same
direction) or flip (opposite direction) open a brand-new trade with FRESH vol-target
sizing and a FRESH ATR-14 stop as of that bar -- but this reentry/flip trade watches ONLY
the ATR stop, no further TP, so it cannot itself trigger a third decision. Flat means no
forced re-entry; the ordinary primary bar-close signal resumes checking as normal on
subsequent bars (exactly today's "flat until next signal" behavior).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m
from tsmom_barrier_sim import extract_market_tuples
from tsmom_walkforward import walk_forward_one
from tsmom_joint_portfolio_barrier_sim import LOCKED, WEIGHT, BLOCK_LEN_DAYS, N_SIMS, CYCLE_CAP_DAYS, simulate_pooled_path
from tsmom_exit_variants import _atr_series, walk_forward_variant, run_barrier_sim_variant, outlier_check, ATR_K

OUT = Path(__file__).parent / "output"

# Fixed per combo, carried over from session 24's H_TP_EXIT train-selected `alt`-mode
# values -- reused here as a stage-1 TRIGGER only, not a re-endorsement of TP-as-
# terminal-exit (that verdict was mostly negative).
TP_ATR_MULT_BY_COMBO = {("BTC", "4h"): 3.5, ("ETH", "8h"): 2.0, ("ETH", "12h"): 3.5, ("SOL", "12h"): 2.5}
ATR_MULT_BASE = 2.5

SCORE_TYPES = ("raw", "vol_norm")
SCORE_LOOKBACK_GRID = [2, 3, 5]
EPS_GRID = {"raw": [0.0, 0.002, 0.005, 0.01], "vol_norm": [0.0, 0.25, 0.5, 1.0]}


def _score_arrays(close: np.ndarray, log_ret_padded: np.ndarray, score_lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """score_mom_log[t] = log(close[t]/close[t-score_lookback]) -- a plain trailing
    log-return over the last score_lookback bars, evaluated AT t (not shifted: by the
    time we're computing this, bar t is fully realized, we're deciding the action for
    t+1 onward, exactly mirroring the base signal's own "decide using data through t,
    act at t+1" convention, just triggered reactively by the TP-fire event instead of on
    every bar). score_local_vol[t] = rolling std of 1-bar log returns over the same
    trailing window ending at t, used to vol-normalize the raw score."""
    close_s = pd.Series(close)
    score_mom_log = np.log(close_s / close_s.shift(score_lookback)).to_numpy()
    score_local_vol = pd.Series(log_ret_padded).rolling(score_lookback).std().to_numpy()
    return score_mom_log, score_local_vol


def vol_target_returns_post_tp(log_ret: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                lookback_bars: int, atr_k: int, coin: str, interval: str,
                                tp_atr_mult: float, score_type: str, score_lookback: int, eps: float,
                                atr_mult: float = ATR_MULT_BASE,
                                round_trip_cost: float = m.ROUND_TRIP_COST) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    mom = pd.Series(close).pct_change(lookback_bars).shift(1).to_numpy()
    atr = _atr_series(high, low, close, atr_k)
    log_ret_padded = np.concatenate([[np.nan], log_ret])
    realized_vol = pd.Series(log_ret_padded).rolling(m.VOL_LOOKBACK_BARS).std().shift(1).to_numpy()
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(m.BARS_PER_YEAR[interval])
    score_mom_log, score_local_vol = _score_arrays(close, log_ret_padded, score_lookback)

    pos_dir_arr = np.zeros(n)
    strat_ret = np.zeros(n)
    pos_dir, stop_level, tp_level, pos_size, has_tp = 0, np.nan, np.nan, 0.0, False
    forced_dir = 0  # set at a TP-fire bar t, consumed when opening a trade at bar t+1

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if forced_dir != 0 and not np.isnan(atr[t]) and not np.isnan(realized_vol[t]) and realized_vol[t] > 0:
                pos_dir = forced_dir
                pos_size = min(target_vol_per_bar / realized_vol[t], m.LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr[t]
                has_tp = False  # no chaining: a re-entry/flip trade watches only its own ATR stop
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * round_trip_cost / 2
            elif (not np.isnan(mom[t]) and mom[t] != 0 and not np.isnan(atr[t])
                    and not np.isnan(realized_vol[t]) and realized_vol[t] > 0):
                pos_dir = 1 if mom[t] > 0 else -1
                pos_size = min(target_vol_per_bar / realized_vol[t], m.LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr[t]
                tp_level = prev_close + pos_dir * tp_atr_mult * atr[t]
                has_tp = True
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * round_trip_cost / 2
            forced_dir = 0
        else:
            stop_hit = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            tp_hit = has_tp and ((high[t] >= tp_level) if pos_dir == 1 else (low[t] <= tp_level))
            if stop_hit or tp_hit:
                exit_px = stop_level if stop_hit else tp_level  # both-in-one-bar: adverse exit wins, conservative
                strat_ret[t] = pos_dir * pos_size * np.log(exit_px / prev_close) - pos_size * round_trip_cost / 2
                if tp_hit and not stop_hit:
                    raw_score = score_mom_log[t]
                    if np.isnan(raw_score):
                        forced_dir = 0
                    else:
                        if score_type == "raw":
                            chosen = raw_score
                        else:
                            local_vol = score_local_vol[t]
                            chosen = raw_score / (local_vol * np.sqrt(score_lookback)) if local_vol > 0 else 0.0
                        # chosen is an ABSOLUTE reading (positive iff price rose over the
                        # score window, regardless of pos_dir) -- must be made relative to
                        # pos_dir before thresholding, or the hold/flip mapping is inverted
                        # for short trades (BUG, caught and fixed here: the original
                        # session-25 code thresholded `chosen` directly, which happened to
                        # be correct for pos_dir=+1 but backwards for pos_dir=-1).
                        signed_chosen = pos_dir * chosen
                        if signed_chosen > eps:
                            forced_dir = pos_dir       # momentum holds -> re-enter same direction
                        elif signed_chosen < -eps:
                            forced_dir = -pos_dir      # reversal -> flip
                        else:
                            forced_dir = 0              # flat until next primary signal
                else:
                    forced_dir = 0  # stop-out (no TP): no second decision, matches production behavior
                pos_dir, stop_level, tp_level, pos_size, has_tp = 0, np.nan, np.nan, 0.0, False
            else:
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr[t]):
                    stop_level = max(stop_level, close[t] - atr_mult * atr[t]) if pos_dir == 1 else min(stop_level, close[t] + atr_mult * atr[t])
        pos_dir_arr[t] = pos_dir
    return pos_dir_arr, strat_ret


def build_daily_series_post_tp(param_star_by_combo: dict) -> pd.DataFrame:
    """Analogous to tsmom_directional_variants.build_daily_series_variant: per-combo OOS
    test-slice return series (bar_range from walk_forward_variant, never touching train
    bars) scaled by the locked k and quarter-weight, resampled to daily, inner-joined --
    feeds the SAME pooled joint-block-bootstrap machinery H_JOINT_PORTFOLIO/H_DIRECTIONAL
    use, for a directly comparable pooled p_pass number."""
    from tsmom_exit_variants import _real_history_series
    cols = {}
    for (coin, interval), (lb, k) in LOCKED.items():
        bar_range, params = param_star_by_combo[(coin, interval)]
        s = _real_history_series(coin, interval, lb, vol_target_returns_post_tp,
                                  tp_atr_mult=TP_ATR_MULT_BY_COMBO[(coin, interval)], **params)
        test = s.iloc[bar_range[0]:bar_range[1]]
        daily = test.resample("1D").sum() * k * WEIGHT
        cols[f"{coin}_{interval}"] = daily
    return pd.DataFrame(cols).dropna(how="any")


if __name__ == "__main__":
    print("=== Sanity check: TP disabled (huge tp_atr_mult) must reproduce vol_target_returns(atr_mult=2.5) bar-for-bar ===")
    from tsmom_barrier_sim import vol_target_returns as _baseline_vtr
    for (coin, interval), (lb, _k) in LOCKED.items():
        log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval)
        close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret_hist))])
        high_full = np.concatenate([[1.0], close_full[1:] * np.exp(hi_hist)])
        low_full = np.concatenate([[1.0], close_full[1:] * np.exp(lo_hist)])
        _, ref = _baseline_vtr(log_ret_hist, high_full, low_full, close_full, lb, ATR_K, ATR_MULT_BASE, coin, interval)
        _, post = vol_target_returns_post_tp(log_ret_hist, high_full, low_full, close_full, lb, ATR_K, coin, interval,
                                              tp_atr_mult=1e6, score_type="raw", score_lookback=2, eps=0.0)
        match = np.allclose(ref, post, atol=1e-10)
        print(f"  {coin} {interval}: {'OK' if match else 'MISMATCH -- fix vol_target_returns_post_tp'}")
        if not match:
            sys.exit("TP-disabled baseline-reproduction sanity check failed.")

    print("\n=== H_POST_TP_REENTRY: per-combo walk-forward (score_type x score_lookback x eps grid) ===")
    rows = []
    param_star_by_combo = {}
    for (coin, interval), (lb, _k) in LOCKED.items():
        tp = TP_ATR_MULT_BY_COMBO[(coin, interval)]
        grid = [{"score_type": st, "score_lookback": lbk, "eps": e}
                 for st in SCORE_TYPES for lbk in SCORE_LOOKBACK_GRID for e in EPS_GRID[st]]
        wf = walk_forward_variant(coin, interval, lb, vol_target_returns_post_tp, grid,
                                   fixed_kwargs={"tp_atr_mult": tp})
        full_sh, drop_sh = outlier_check(coin, interval, lb, vol_target_returns_post_tp,
                                          wf["param_star"], {"tp_atr_mult": tp})
        rows.append({"coin": coin, "interval": interval, "tp_atr_mult": tp, **wf,
                     "full_sharpe": full_sh, "outlier_dropped_sharpe": drop_sh})
        param_star_by_combo[(coin, interval)] = (wf["bar_range"], wf["param_star"])

    df = pd.DataFrame(rows)
    fmt = {"train_sharpe_at_star": "{:.2f}".format, "test_sharpe": "{:.2f}".format,
           "test_max_dd": "{:.1%}".format, "test_total_return": "{:.1%}".format,
           "oos_p_pass": "{:.1%}".format, "full_sharpe": "{:.2f}".format,
           "outlier_dropped_sharpe": "{:.2f}".format}
    print(df[["coin", "interval", "tp_atr_mult", "param_star", "train_sharpe_at_star", "test_sharpe",
              "test_max_dd", "oos_p_pass", "full_sharpe", "outlier_dropped_sharpe"]].to_string(index=False, formatters=fmt))

    print("\n=== Reference: current locked baseline (session 11 walk-forward, tsmom_walkforward.json) ===")
    base = pd.read_json(OUT / "tsmom_walkforward.json")
    base = base[base.apply(lambda r: (r["coin"], r["interval"]) in LOCKED, axis=1)]
    print(base[["coin", "interval", "lb_star", "test_sharpe_at_lb_star", "test_max_dd", "oos_p_pass"]]
          .to_string(index=False, formatters={"test_sharpe_at_lb_star": "{:.2f}".format,
                                                "test_max_dd": "{:.1%}".format, "oos_p_pass": "{:.1%}".format}))

    print("\n=== Portfolio-level benchmark: pooled p_pass vs. H_JOINT_PORTFOLIO's baseline (81.0% full / 63.6% ex-rally) ===")
    daily_df = build_daily_series_post_tp(param_star_by_combo)
    print(f"Common calendar overlap: {daily_df.index.min().date()} -> {daily_df.index.max().date()}, n={len(daily_df)} days")
    print("Pairwise correlation of quarter-weighted daily strategy returns:")
    print(daily_df.corr().round(2).to_string())
    combined_hist = daily_df.sum(axis=1)
    pooled_sharpe = combined_hist.mean() / combined_hist.std() * np.sqrt(365) if combined_hist.std() > 0 else np.nan
    print(f"Pooled daily return: mean={combined_hist.mean():.5f}, std={combined_hist.std():.5f}, ann. Sharpe={pooled_sharpe:.2f}")

    daily_mat = daily_df.to_numpy()
    rng = np.random.default_rng(20260907)  # same seed as H_JOINT_PORTFOLIO/H_DIRECTIONAL for comparable draws
    outcomes = [simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng) for _ in range(N_SIMS)]
    names = [o[0] for o in outcomes]
    for name in ["pass", "fail_static", "fail_daily", "unresolved"]:
        print(f"  {name:>12}: {names.count(name)/N_SIMS:.1%}")

    EPISODE = ("2026-08-17", "2026-08-27")
    mask = ~((daily_df.index >= EPISODE[0]) & (daily_df.index <= EPISODE[1]))
    ex = daily_df.loc[mask]
    ex_combined = ex.sum(axis=1)
    ex_sharpe = ex_combined.mean() / ex_combined.std() * np.sqrt(365) if ex_combined.std() > 0 else np.nan
    rng2 = np.random.default_rng(20260907)
    ex_outcomes = [simulate_pooled_path(ex.to_numpy(), BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng2) for _ in range(N_SIMS)]
    ex_names = [o[0] for o in ex_outcomes]
    print(f"\nEx Aug 17-27 rally (n={len(ex)} days): ann. Sharpe={ex_sharpe:.2f}, "
          f"p_pass={ex_names.count('pass')/N_SIMS:.1%}, fail_static={ex_names.count('fail_static')/N_SIMS:.1%}")

    OUT.mkdir(exist_ok=True)
    df.drop(columns=["param_star"]).assign(param_star=df["param_star"].astype(str)).to_json(
        OUT / "tsmom_post_tp_reentry.json", orient="records", indent=2)
