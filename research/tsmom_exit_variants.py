"""
H_TP_EXIT / H_WIDE_TRAIL: two exit-mechanism variants (session 24), re-derived and
properly documented after a full search of this repo found no trace of either ever being
tested before, despite a recollection that a capped take-profit was tried and killed and
a "let runners run" wider trailing stop was found to be a false positive on more rigorous
testing. Neither claim survived a full-text search of `research/` (see H_GIVEBACK's
session-23 preamble and PROJECT.md session 23) -- the current ATR-2.5x trailing stop was
in fact validated (session 11's walk-forward) as this project's most robust result. This
session tests both ideas honestly, from scratch, informed by H_GIVEBACK's (session 23)
empirical MFE distribution rather than blind guesses.

Both variants keep the entry signal exactly as-is (sign of trailing return, vol-target
sizing) and only change the exit rule, following the SAME causal-timing discipline as
`tsmom_barrier_sim.vol_target_returns` (level fixed using data through t-1, breach
checked against bar t's own intrabar high/low, return truncated to the exit price) --
the exact discipline session 10's look-ahead bug violated, so it gets the same explicit
vigilance here. Both operate on the same (log_ret, high, low, close) numpy-array
convention `vol_target_returns` uses, so the SAME function serves both a real-history
walk-forward evaluation (arrays reconstructed from `extract_market_tuples`, matching
`tsmom_barrier_sim.py`'s own `__main__` convention -- a uniformly rescaled synthetic price
path starting at 1.0 preserves the strategy's dynamics exactly, since every quantity the
loop uses is either a ratio/log-ratio or an ATR-proportional offset) and Monte Carlo
bootstrap paths, instead of needing two separate implementations.

(a) H_TP_EXIT: a take-profit level fixed at entry (entry_px +/- tp_atr_mult * ATR at
entry), expressed in ATR-multiples (not raw percent) for unit consistency with the
existing stop and because H_GIVEBACK found typical MFE magnitude varies enormously by
combo in raw percent terms (BTC 4h median ~2.8%, SOL 12h median ~13.1%) but ATR already
normalizes for that. tp_mode="alt": TP and the existing ATR stop both live, whichever
triggers first exits. tp_mode="tp_only": ATR stop disabled, TP is the sole exit. If both
trigger within the same bar (only possible for "alt" mode), the adverse (stop) exit is
assumed to resolve first -- a deliberately conservative convention given OHLC data alone
cannot disambiguate genuine intrabar sequencing, avoiding the kind of look-ahead-flavored
optimism session 10's bug produced.

(b) H_WIDE_TRAIL: either a simple constant wider ATR multiplier (a direct parameter to
the existing, unmodified `vol_target_returns` -- no new code needed for that sub-case) or
a stepped/ratcheting schedule that widens the multiplier once favorable excursion (in
ATR-units from entry, using the CURRENT bar's own ATR, same denominator convention the
stop's own offset uses) crosses a threshold. A schedule of exactly `[(0.0, 2.5)]`
reproduces the current production stop bar-for-bar -- used below as a sanity check, not
just a candidate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m
from tsmom_barrier_sim import (
    extract_market_tuples, block_bootstrap_path, vol_target_returns,
    TARGET, STATIC_FLOOR, DAILY_LOSS, BLOCK_LEN,
)
from tsmom_joint_portfolio_barrier_sim import LOCKED

OUT = Path(__file__).parent / "output"
TRAIN_FRAC = 0.7
ATR_K = 14
ATR_MULT_BASE = 2.5
N_PATHS = 400
HORIZON_DAYS = 1095
BARS_PER_DAY = {"4h": 6, "8h": 3, "12h": 2}

TP_ATR_GRID = [1.0, 1.5, 2.0, 2.5, 3.5]
WIDE_FIXED_GRID = [3.5, 4.0, 5.0]
RATCHET_CANDIDATES = {
    # name -> schedule: sorted list of (favorable_excursion_atr_threshold, atr_mult)
    "fixed_2.5_baseline": [(0.0, ATR_MULT_BASE)],  # == current production, sanity-check row
    "fixed_3.5": [(0.0, 3.5)],
    "fixed_4.0": [(0.0, 4.0)],
    "fixed_5.0": [(0.0, 5.0)],
    "ratchet_2.5->4.0@2atr": [(0.0, 2.5), (2.0, 4.0)],
    "ratchet_2.5->5.0@3atr": [(0.0, 2.5), (3.0, 5.0)],
}


def _atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_k: int) -> np.ndarray:
    tr = np.concatenate([[np.nan], np.maximum(
        high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))])
    return pd.Series(tr).rolling(atr_k).mean().shift(1).to_numpy()


def vol_target_returns_with_tp(log_ret: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                lookback_bars: int, atr_k: int, atr_mult: float, tp_atr_mult: float,
                                coin: str, interval: str, tp_mode: str = "alt",
                                round_trip_cost: float = m.ROUND_TRIP_COST) -> tuple[np.ndarray, np.ndarray]:
    """Same entry/sizing as vol_target_returns, exit is the first of {ATR stop
    (disabled if tp_mode=='tp_only'), TP (fixed at entry, never ratchets)} to trigger.
    Both checked against bar t's own intrabar high/low, level fixed using data through
    t-1 only. Returns (pos_dir_arr, strat_ret)."""
    n = len(close)
    mom = pd.Series(close).pct_change(lookback_bars).shift(1).to_numpy()
    atr = _atr_series(high, low, close, atr_k)
    log_ret_padded = np.concatenate([[np.nan], log_ret])
    realized_vol = pd.Series(log_ret_padded).rolling(m.VOL_LOOKBACK_BARS).std().shift(1).to_numpy()
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(m.BARS_PER_YEAR[interval])

    pos_dir_arr = np.zeros(n)
    strat_ret = np.zeros(n)
    pos_dir, stop_level, tp_level, pos_size = 0, np.nan, np.nan, 0.0

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if (not np.isnan(mom[t]) and mom[t] != 0 and not np.isnan(atr[t])
                    and not np.isnan(realized_vol[t]) and realized_vol[t] > 0):
                pos_dir = 1 if mom[t] > 0 else -1
                pos_size = min(target_vol_per_bar / realized_vol[t], m.LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr[t]
                tp_level = prev_close + pos_dir * tp_atr_mult * atr[t]
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * round_trip_cost / 2
        else:
            stop_hit = False if tp_mode == "tp_only" else (
                (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level))
            tp_hit = (high[t] >= tp_level) if pos_dir == 1 else (low[t] <= tp_level)
            if stop_hit or tp_hit:
                # both-in-one-bar is ambiguous from OHLC alone -- conservatively resolve
                # the adverse exit (stop) first, never the optimistic one
                exit_px = stop_level if stop_hit else tp_level
                strat_ret[t] = pos_dir * pos_size * np.log(exit_px / prev_close) - pos_size * round_trip_cost / 2
                pos_dir, stop_level, tp_level, pos_size = 0, np.nan, np.nan, 0.0
            else:
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr[t]):
                    stop_level = max(stop_level, close[t] - atr_mult * atr[t]) if pos_dir == 1 else min(stop_level, close[t] + atr_mult * atr[t])
                # tp_level intentionally not ratcheted -- a capped TP is fixed by definition
        pos_dir_arr[t] = pos_dir
    return pos_dir_arr, strat_ret


def vol_target_returns_ratchet(log_ret: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                lookback_bars: int, atr_k: int, coin: str, interval: str,
                                ratchet_schedule: list[tuple[float, float]],
                                round_trip_cost: float = m.ROUND_TRIP_COST) -> tuple[np.ndarray, np.ndarray]:
    """Same as vol_target_returns but the ATR multiplier used to compute each bar's
    candidate stop level depends on the trade's current favorable excursion (in
    ATR-units from entry, using the CURRENT bar's own ATR as denominator): the largest
    ratchet_schedule threshold <= current excursion selects that tier's atr_mult.
    ratchet_schedule must be sorted ascending by threshold and start at 0.0. The stop
    level itself still only ever ratchets favorably (max/min against the prior level),
    so a wider tier that would imply a LOOSER stop than already locked in simply fails
    to advance rather than retreating. Returns (pos_dir_arr, strat_ret)."""
    n = len(close)
    mom = pd.Series(close).pct_change(lookback_bars).shift(1).to_numpy()
    atr = _atr_series(high, low, close, atr_k)
    log_ret_padded = np.concatenate([[np.nan], log_ret])
    realized_vol = pd.Series(log_ret_padded).rolling(m.VOL_LOOKBACK_BARS).std().shift(1).to_numpy()
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(m.BARS_PER_YEAR[interval])
    thresholds = np.array([s[0] for s in ratchet_schedule])
    mults = np.array([s[1] for s in ratchet_schedule])

    pos_dir_arr = np.zeros(n)
    strat_ret = np.zeros(n)
    pos_dir, stop_level, pos_size, entry_px = 0, np.nan, 0.0, np.nan

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if (not np.isnan(mom[t]) and mom[t] != 0 and not np.isnan(atr[t])
                    and not np.isnan(realized_vol[t]) and realized_vol[t] > 0):
                pos_dir = 1 if mom[t] > 0 else -1
                pos_size = min(target_vol_per_bar / realized_vol[t], m.LEVERAGE_CAP[coin])
                entry_px = prev_close
                stop_level = prev_close - pos_dir * mults[0] * atr[t]
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * round_trip_cost / 2
        else:
            breached = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            if breached:
                strat_ret[t] = pos_dir * pos_size * np.log(stop_level / prev_close) - pos_size * round_trip_cost / 2
                pos_dir, stop_level, pos_size, entry_px = 0, np.nan, 0.0, np.nan
            else:
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr[t]) and atr[t] > 0:
                    excursion_atr = pos_dir * (close[t] - entry_px) / atr[t]
                    tier = np.searchsorted(thresholds, excursion_atr, side="right") - 1
                    tier = max(0, tier)
                    active_mult = mults[tier]
                    stop_level = max(stop_level, close[t] - active_mult * atr[t]) if pos_dir == 1 else min(stop_level, close[t] + active_mult * atr[t])
        pos_dir_arr[t] = pos_dir
    return pos_dir_arr, strat_ret


def _real_history_series(coin: str, interval: str, lookback_bars: int, variant_fn, **kwargs) -> pd.Series:
    """Builds a datetime-indexed strat_ret series for `variant_fn` on the FULL real
    history (bar_range=None), matching _strat_ret_series' index convention exactly, so
    it can be sliced by the same train/test split every other walk-forward uses. Reuses
    extract_market_tuples + the uniformly-rescaled synthetic-price reconstruction
    tsmom_barrier_sim.py's own __main__ already relies on for real-history evaluation."""
    log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval)
    close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret_hist))])
    high_full = np.concatenate([[1.0], close_full[1:] * np.exp(hi_hist)])
    low_full = np.concatenate([[1.0], close_full[1:] * np.exp(lo_hist)])
    _, strat_ret = variant_fn(log_ret_hist, high_full, low_full, close_full, lookback_bars,
                              ATR_K, coin=coin, interval=interval, **kwargs)
    df = m.load_candles(coin, interval)
    return pd.Series(strat_ret, index=df.index).iloc[1:]


def _sharpe(s: pd.Series, bars_per_year: int) -> float:
    if len(s) < 2 or s.std() == 0:
        return np.nan
    return (s.mean() / s.std()) * np.sqrt(bars_per_year)


def walk_forward_variant(coin: str, interval: str, lookback_bars: int, variant_fn, param_grid: list[dict],
                          fixed_kwargs: dict | None = None) -> dict:
    """Structurally parallel to tsmom_walkforward.walk_forward_one: for each candidate
    param dict in param_grid, build the real-history series, split 70/30, select the
    param with the best TRAIN Sharpe, report TEST Sharpe/max_dd/total_return, then run a
    barrier Monte Carlo restricted to the test-only bar range under that selected
    param."""
    fixed_kwargs = fixed_kwargs or {}
    bars_per_year = m.BARS_PER_YEAR[interval]
    series_by_param = {i: _real_history_series(coin, interval, lookback_bars, variant_fn, **{**fixed_kwargs, **p})
                        for i, p in enumerate(param_grid)}
    n_bars_series = len(next(iter(series_by_param.values())))
    split = int(n_bars_series * TRAIN_FRAC)

    train_sharpe = {i: _sharpe(s.iloc[:split], bars_per_year) for i, s in series_by_param.items()}
    i_star = max(train_sharpe, key=lambda k: (train_sharpe[k] if not np.isnan(train_sharpe[k]) else -np.inf))
    param_star = param_grid[i_star]

    test_series = series_by_param[i_star].iloc[split:]
    test_sharpe = _sharpe(test_series, bars_per_year)
    test_equity = (1 + test_series).cumprod()
    test_dd = (test_equity / test_equity.cummax() - 1).min()
    test_total_return = test_equity.iloc[-1] - 1

    bar_range = (split, n_bars_series)
    sim = run_barrier_sim_variant(coin, interval, lookback_bars, variant_fn, param_star, fixed_kwargs,
                                   bar_range=bar_range, n_paths=N_PATHS)

    return {
        "coin": coin, "interval": interval, "param_star": param_star,
        "train_sharpe_at_star": train_sharpe[i_star], "test_sharpe": test_sharpe,
        "test_max_dd": test_dd, "test_total_return": test_total_return,
        "test_n_bars": n_bars_series - split, "bar_range": bar_range,
        "oos_p_pass": sim["pass"], "oos_fail_static": sim["fail_static"],
        "oos_fail_daily": sim["fail_daily"], "oos_unresolved": sim["unresolved"],
    }


def simulate_path_variant(coin: str, interval: str, lookback_bars: int, variant_fn, params: dict,
                           fixed_kwargs: dict, n_bars: int, log_ret_hist, hi_hist, lo_hist,
                           rng: np.random.Generator, round_trip_cost: float = m.ROUND_TRIP_COST) -> str:
    """Parallel duplicate of tsmom_barrier_sim.simulate_path for an arbitrary exit
    variant function -- same block-bootstrap/barrier-check structure, fixed at
    leverage=1.0 (the plain vol-targeted process with no MI&A/CPPI overlay, the
    project's own primary comparison baseline -- session 14's corrected k=1 numbers),
    since this is a brand-new, not-yet-baseline-validated mechanism and layering
    untested sizing overlays on top of it would conflate two separate open questions."""
    # Same (log_ret, high_full, low_full, close_full) construction _real_history_series
    # uses for real data -- high_full[t]/low_full[t]/close_full[t] all refer to the SAME
    # bar t (length n_bars+1), matching what variant_fn (vol_target_returns's own
    # array-index convention) expects. This differs from tsmom_barrier_sim.simulate_path's
    # own high/low offset convention (built for its *different*, off-by-one indexing
    # against `close`), so is deliberately not reused as-is here.
    block_len = BLOCK_LEN[interval]
    log_ret, high_frac, low_frac = block_bootstrap_path(log_ret_hist, hi_hist, lo_hist, n_bars, block_len, rng)
    close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret))])
    high_full = np.concatenate([[1.0], close_full[1:] * np.exp(high_frac)])
    low_full = np.concatenate([[1.0], close_full[1:] * np.exp(low_frac)])

    _, strat_ret = variant_fn(log_ret, high_full, low_full, close_full, lookback_bars,
                              ATR_K, coin=coin, interval=interval, round_trip_cost=round_trip_cost,
                              **{**fixed_kwargs, **params})

    bars_per_day = BARS_PER_DAY[interval]
    equity, day_open = 1.0, 1.0
    for t in range(1, len(close_full)):
        if (t - 1) % bars_per_day == 0:
            day_open = equity
        equity *= np.exp(strat_ret[t])
        if equity <= 1 + STATIC_FLOOR:
            return "fail_static"
        if equity <= day_open * (1 - DAILY_LOSS):
            return "fail_daily"
        if equity >= 1 + TARGET:
            return "pass"
    return "unresolved"


def run_barrier_sim_variant(coin: str, interval: str, lookback_bars: int, variant_fn, params: dict,
                             fixed_kwargs: dict, bar_range: tuple[int, int] | None = None,
                             n_paths: int = N_PATHS, horizon_days: int = HORIZON_DAYS, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval, bar_range)
    n_bars = int(horizon_days * 24 / int(interval[:-1]))
    outcomes = {"pass": 0, "fail_static": 0, "fail_daily": 0, "unresolved": 0}
    for _ in range(n_paths):
        r = simulate_path_variant(coin, interval, lookback_bars, variant_fn, params, fixed_kwargs,
                                   n_bars, log_ret_hist, hi_hist, lo_hist, rng)
        outcomes[r] += 1
    return {k: v / n_paths for k, v in outcomes.items()}


def outlier_check(coin: str, interval: str, lookback_bars: int, variant_fn, params: dict,
                   fixed_kwargs: dict, n_drop: int = 20) -> tuple[float, float]:
    """Bar-level drop-N-largest-magnitude-bars check, matching H_TSMOM's own convention
    -- returns (full_sharpe, dropped_sharpe)."""
    s = _real_history_series(coin, interval, lookback_bars, variant_fn, **{**fixed_kwargs, **params})
    bars_per_year = m.BARS_PER_YEAR[interval]
    full = _sharpe(s, bars_per_year)
    keep = s.abs().sort_values(ascending=False).index[n_drop:]
    dropped = _sharpe(s.loc[s.index.isin(keep)].sort_index(), bars_per_year)
    return full, dropped


if __name__ == "__main__":
    print("=== Sanity check 1: ratchet_schedule=[(0.0, 2.5)] must reproduce vol_target_returns(atr_mult=2.5) bar-for-bar ===")
    for (coin, interval), (lb, _k) in LOCKED.items():
        log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval)
        close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret_hist))])
        high_full = np.concatenate([[1.0], close_full[1:] * np.exp(hi_hist)])
        low_full = np.concatenate([[1.0], close_full[1:] * np.exp(lo_hist)])
        _, ref = vol_target_returns(log_ret_hist, high_full, low_full, close_full, lb, ATR_K, ATR_MULT_BASE, coin, interval)
        _, rat = vol_target_returns_ratchet(log_ret_hist, high_full, low_full, close_full, lb, ATR_K, coin, interval,
                                             RATCHET_CANDIDATES["fixed_2.5_baseline"])
        match = np.allclose(ref, rat, atol=1e-10)
        print(f"  {coin} {interval}: {'OK' if match else 'MISMATCH -- fix vol_target_returns_ratchet'}")
        if not match:
            sys.exit("Ratchet baseline-reproduction sanity check failed.")

    print("\n=== Sanity check 2: TP disabled (huge tp_atr_mult, mode='alt') must reproduce vol_target_returns(atr_mult=2.5) ===")
    for (coin, interval), (lb, _k) in LOCKED.items():
        log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval)
        close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret_hist))])
        high_full = np.concatenate([[1.0], close_full[1:] * np.exp(hi_hist)])
        low_full = np.concatenate([[1.0], close_full[1:] * np.exp(lo_hist)])
        _, ref = vol_target_returns(log_ret_hist, high_full, low_full, close_full, lb, ATR_K, ATR_MULT_BASE, coin, interval)
        _, tp = vol_target_returns_with_tp(log_ret_hist, high_full, low_full, close_full, lb, ATR_K, ATR_MULT_BASE,
                                            1e6, coin, interval, tp_mode="alt")
        match = np.allclose(ref, tp, atol=1e-10)
        print(f"  {coin} {interval}: {'OK' if match else 'MISMATCH -- fix vol_target_returns_with_tp'}")
        if not match:
            sys.exit("TP-disabled baseline-reproduction sanity check failed.")

    print("\n=== H_TP_EXIT: walk-forward, both tp_mode variants, ATR-multiple grid ===")
    tp_rows = []
    for (coin, interval), (lb, _k) in LOCKED.items():
        for tp_mode in ("alt", "tp_only"):
            grid = [{"tp_atr_mult": g, "atr_mult": ATR_MULT_BASE, "tp_mode": tp_mode} for g in TP_ATR_GRID]
            wf = walk_forward_variant(coin, interval, lb, vol_target_returns_with_tp, grid)
            full_sh, drop_sh = outlier_check(coin, interval, lb, vol_target_returns_with_tp,
                                              wf["param_star"], {})
            tp_rows.append({"coin": coin, "interval": interval, "tp_mode": tp_mode, **wf,
                             "full_sharpe": full_sh, "outlier_dropped_sharpe": drop_sh})
    tp_df = pd.DataFrame(tp_rows)
    fmt = {"train_sharpe_at_star": "{:.2f}".format, "test_sharpe": "{:.2f}".format,
           "test_max_dd": "{:.1%}".format, "test_total_return": "{:.1%}".format,
           "oos_p_pass": "{:.1%}".format, "oos_fail_static": "{:.1%}".format,
           "oos_fail_daily": "{:.1%}".format, "oos_unresolved": "{:.1%}".format,
           "full_sharpe": "{:.2f}".format, "outlier_dropped_sharpe": "{:.2f}".format}
    print(tp_df[["coin", "interval", "tp_mode", "param_star", "train_sharpe_at_star", "test_sharpe",
                 "test_max_dd", "oos_p_pass", "full_sharpe", "outlier_dropped_sharpe"]].to_string(index=False, formatters=fmt))

    print("\n=== H_WIDE_TRAIL: walk-forward over fixed-wider and ratchet schedules ===")
    wide_rows = []
    for (coin, interval), (lb, _k) in LOCKED.items():
        grid = [{"ratchet_schedule": sched} for sched in RATCHET_CANDIDATES.values()]
        wf = walk_forward_variant(coin, interval, lb, vol_target_returns_ratchet, grid)
        name_by_sched = {str(v): k for k, v in RATCHET_CANDIDATES.items()}
        star_name = name_by_sched[str(wf["param_star"]["ratchet_schedule"])]
        full_sh, drop_sh = outlier_check(coin, interval, lb, vol_target_returns_ratchet, wf["param_star"], {})
        wide_rows.append({"coin": coin, "interval": interval, "selected_schedule": star_name, **wf,
                           "full_sharpe": full_sh, "outlier_dropped_sharpe": drop_sh})
    wide_df = pd.DataFrame(wide_rows)
    print(wide_df[["coin", "interval", "selected_schedule", "train_sharpe_at_star", "test_sharpe",
                   "test_max_dd", "oos_p_pass", "full_sharpe", "outlier_dropped_sharpe"]].to_string(index=False, formatters=fmt))

    print("\n=== Reference: current locked baseline (session 11 walk-forward numbers, tsmom_walkforward.json) ===")
    try:
        base = pd.read_json(OUT / "tsmom_walkforward.json")
        print(base[["coin", "interval", "lb_star", "test_sharpe_at_lb_star", "test_max_dd", "oos_p_pass"]]
              .to_string(index=False, formatters={"test_sharpe_at_lb_star": "{:.2f}".format,
                                                    "test_max_dd": "{:.1%}".format, "oos_p_pass": "{:.1%}".format}))
    except FileNotFoundError:
        print("  (tsmom_walkforward.json not found -- run tsmom_walkforward.py first for a saved baseline reference)")

    OUT.mkdir(exist_ok=True)
    tp_df.drop(columns=["param_star"]).assign(param_star=tp_df["param_star"].astype(str)).to_json(
        OUT / "tsmom_tp_exit.json", orient="records", indent=2)
    wide_df.drop(columns=["param_star"]).assign(param_star=wide_df["param_star"].astype(str)).to_json(
        OUT / "tsmom_wide_trail.json", orient="records", indent=2)
