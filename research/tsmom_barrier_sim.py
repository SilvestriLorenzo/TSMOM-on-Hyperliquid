"""
H_TSMOM, translated into an actual Propr p_pass (session 10, continuing
tsmom_hypothesis.py). Three sizing regimes compared:
  A) fixed vol-target (what tsmom_hypothesis.py already tested)
  B) Magdon-Ismail & Atiya-calibrated fixed leverage, targeting E[max drawdown] = 1.5%
     over a 1095-bar-equivalent horizon (see mi_atiya_drawdown.py)
  C) B plus Grossman-Zhou-style dynamic de-risking: exposure scaled down as equity
     approaches either Propr barrier (the static floor from the starting balance, and
     the separate daily-loss floor that resets each day) -- adapted from Grossman-Zhou's
     classic trailing-high-water-mark framework to Propr's actual FIXED floor, which is
     a simpler case (no ratchet needed for the static barrier; the daily-loss barrier is
     the one genuinely short-horizon trailing constraint, reset each day).

Why a fresh per-path simulation is needed, not a resample-and-scale of the existing
backtest: the ATR-trailing-stop signal is genuinely path-dependent (positions and stop
levels evolve with the realized price path). Resampling the strategy's own historical
RETURNS would inherit whatever specific price path happened to produce them; instead,
this block-bootstraps the raw (log_ret, high_frac, low_frac) market-data tuples
(preserving local trend/volatility clustering, contiguous blocks, matching the
methodology arXiv:2602.11708 itself uses for its own significance tests), reconstructs
a synthetic OHLC path from each resampled block sequence, and RE-RUNS the entry+ATR-stop
signal generation fresh on that synthetic path -- exactly as it would run on real,
never-seen data. Propr's barriers are checked directly against the resulting bar-level
equity path (real granularity, not the Brownian-bridge approximation intraday_barrier.py
needed for daily-close-only data).

CORRECTION (session 14): sessions 10-13 sized every Monte Carlo path with a CONSTANT
leverage multiplier (no per-bar vol adjustment) -- a stand-in adopted so Magdon-Ismail &
Atiya's closed-form (which assumes constant drift/vol) applied directly. That stand-in
was quietly a DIFFERENT, weaker strategy than the one actually validated: on BTC 4h's
walk-forward test slice, the real per-bar vol-targeted process has Sharpe 1.95 vs. 1.43
for the constant-leverage version of the identical entry/exit signal (vol-targeting
improves the signal-to-noise ratio of the trade sequence, a standard trend-following
effect -- not noise). Every session 10-13 p_pass number was therefore computed against
the wrong process. Fixed here: `simulate_path` now runs genuine per-bar vol-targeting
(position size = min(target_vol_per_bar / trailing_realized_vol, leverage_cap), fixed at
entry, exactly matching tsmom_hypothesis.py's actually-backtested rule) on every
synthetic path, with `leverage` repurposed as a SCALAR MULTIPLIER applied on top of that
per-bar sizing (k=1.0 reproduces the plain validated strategy with no overlay; MI&A/CPPI
scale k up or down from there) -- this is also a more faithful reading of the user's
original session-10 proposal (MI&A sizing sizing a vol-targeted TSMOM, not replacing its
vol-targeting). `calibrate_mia_leverage` now estimates mu/sigma from this real
vol-targeted (k=1) process via `vol_target_returns`, not the old constant-unit-size
`unit_leverage_returns` (removed).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_hypothesis import load_candles, ROUND_TRIP_COST, LEVERAGE_CAP, BARS_PER_YEAR, VOL_LOOKBACK_BARS, TARGET_VOL_ANNUAL
from mi_atiya_drawdown import calibrate_leverage_for_target_mdd

TARGET = 0.09
STATIC_FLOOR = -0.03
DAILY_LOSS = 0.03
BARS_PER_DAY = {"4h": 6, "8h": 3, "12h": 2}
HORIZON_BARS_1095D = {"4h": 1095 * 6, "8h": 1095 * 3, "12h": 1095 * 2}
BLOCK_LEN = {"4h": 30, "8h": 20, "12h": 15}  # bars per bootstrap block, ~ a few trade-durations


def extract_market_tuples(coin: str, interval: str, bar_range: tuple[int, int] | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """bar_range, if given, is an (start, end) slice applied AFTER dropping the first
    (NaN-return) row -- i.e. indices into the same 0-based space callers use to slice
    a train/test split of the aligned strategy-return series. None (default) keeps the
    original full-history behavior."""
    df = load_candles(coin, interval)
    log_ret = np.log(df["c"] / df["c"].shift(1))
    high_frac = np.log(df["h"] / df["c"])
    low_frac = np.log(df["l"] / df["c"])
    valid = log_ret.notna()
    log_ret_a = log_ret[valid].to_numpy()
    high_a = high_frac[valid].to_numpy()
    low_a = low_frac[valid].to_numpy()
    if bar_range is not None:
        start, end = bar_range
        log_ret_a, high_a, low_a = log_ret_a[start:end], high_a[start:end], low_a[start:end]
    return log_ret_a, high_a, low_a


def block_bootstrap_path(log_ret: np.ndarray, high_frac: np.ndarray, low_frac: np.ndarray,
                          n_bars: int, block_len: int, rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circular block bootstrap: concatenate random contiguous blocks until n_bars is
    covered, preserving local trend/volatility structure within each block."""
    n_hist = len(log_ret)
    out_ret, out_hi, out_lo = [], [], []
    covered = 0
    while covered < n_bars:
        start = rng.integers(0, n_hist)
        idx = (start + np.arange(block_len)) % n_hist  # circular wrap
        out_ret.append(log_ret[idx])
        out_hi.append(high_frac[idx])
        out_lo.append(low_frac[idx])
        covered += block_len
    return (np.concatenate(out_ret)[:n_bars], np.concatenate(out_hi)[:n_bars],
            np.concatenate(out_lo)[:n_bars])


def vol_target_returns(log_ret: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        lookback_bars: int, atr_k: int, atr_mult: float, coin: str, interval: str,
                        round_trip_cost: float = ROUND_TRIP_COST) -> tuple[np.ndarray, np.ndarray]:
    """Runs the entry+ATR-trailing-stop signal with genuine per-bar vol-targeted sizing
    at leverage multiplier k=1 (position size = min(target_vol_per_bar / trailing
    20-bar realized vol, leverage cap), fixed at entry, held through the trade) on one
    price path -- this is the actually-validated strategy (matches
    tsmom_hypothesis.py's backtest_trailing_stop exactly), replacing session 10-13's
    constant-unit-size approximation. Returns (position_dir, strat_ret)."""
    n = len(close)
    mom = pd.Series(close).pct_change(lookback_bars).shift(1).to_numpy()
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = np.nan
    atr = pd.Series(tr).rolling(atr_k).mean().shift(1).to_numpy()
    log_ret_padded = np.concatenate([[np.nan], log_ret])
    realized_vol = pd.Series(log_ret_padded).rolling(VOL_LOOKBACK_BARS).std().shift(1).to_numpy()
    target_vol_per_bar = TARGET_VOL_ANNUAL / np.sqrt(BARS_PER_YEAR[interval])

    pos_dir_arr = np.zeros(n)
    strat_ret = np.zeros(n)
    pos_dir, stop_level, pos_size = 0, np.nan, 0.0

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if (not np.isnan(mom[t]) and mom[t] != 0 and not np.isnan(atr[t])
                    and not np.isnan(realized_vol[t]) and realized_vol[t] > 0):
                pos_dir = 1 if mom[t] > 0 else -1
                pos_size = min(target_vol_per_bar / realized_vol[t], LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr[t]
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * round_trip_cost / 2
        else:
            breached = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            if breached:
                strat_ret[t] = pos_dir * pos_size * np.log(stop_level / prev_close) - pos_size * round_trip_cost / 2
                pos_dir, stop_level, pos_size = 0, np.nan, 0.0
            else:
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr[t]):
                    stop_level = max(stop_level, close[t] - atr_mult * atr[t]) if pos_dir == 1 else min(stop_level, close[t] + atr_mult * atr[t])
        pos_dir_arr[t] = pos_dir
    return pos_dir_arr, strat_ret


def calibrate_mia_leverage(coin: str, interval: str, lookback_bars: int, atr_k: int,
                            atr_mult: float, target_mdd: float = 0.015,
                            bar_range: tuple[int, int] | None = None,
                            horizon_days: int = 1095,
                            round_trip_cost: float = ROUND_TRIP_COST) -> float:
    """Uses the REAL historical path (not a bootstrap draw, optionally restricted to
    bar_range -- e.g. a walk-forward test slice) to estimate the k=1 vol-targeted
    process's mu, sigma (via vol_target_returns -- the actually-validated strategy, not
    a constant-size stand-in), then solves for the multiplier k targeting
    E[MDD]=target_mdd over a horizon_days-equivalent horizon (default 1095, the
    original convention), capped at Propr's per-asset leverage limit."""
    log_ret, high_frac, low_frac = extract_market_tuples(coin, interval, bar_range)
    close = np.concatenate([[1.0], np.exp(np.cumsum(log_ret))])
    high = np.concatenate([[1.0], close[1:] * np.exp(high_frac)])
    low = np.concatenate([[1.0], close[1:] * np.exp(low_frac)])
    _, strat_ret = vol_target_returns(log_ret, high, low, close, lookback_bars, atr_k, atr_mult,
                                       coin, interval, round_trip_cost=round_trip_cost)
    # mu, sigma must be per CALENDAR bar (flat bars included as zero return), matching
    # the units of T below (also calendar bars) -- filtering to active-only bars here
    # would double-count the strategy's flat-time as if it compounded risk continuously.
    mu, sigma = strat_ret.mean(), strat_ret.std(ddof=1)
    horizon_bars = int(horizon_days * 24 / int(interval[:-1]))
    k = calibrate_leverage_for_target_mdd(mu, sigma, horizon_bars, target_mdd)
    return min(k, LEVERAGE_CAP[coin]), mu, sigma


def simulate_path(coin: str, interval: str, lookback_bars: int, atr_k: float, atr_mult: float,
                   leverage: float, cppi: bool, n_bars: int, log_ret_hist, hi_hist, lo_hist,
                   rng: np.random.Generator, cppi_budget_c: float | None = None,
                   round_trip_cost: float = ROUND_TRIP_COST,
                   return_bars: bool = False) -> str | tuple[str, int]:
    """One Monte Carlo path: block-bootstrap synthetic market data, re-run the signal
    fresh with genuine per-bar vol-targeted sizing (fixed at entry, matching
    tsmom_hypothesis.py's actually-validated rule -- see module docstring), scale by
    `leverage` (a constant multiplier k on top of that vol-targeted size, k=1.0
    reproduces the plain validated strategy with no overlay) and, optionally, a CPPI-
    style cushion ratio, then check Propr's two barriers against the resulting bar path.
    Returns 'pass', 'fail_static', 'fail_daily', or 'unresolved'.

    Sizing modes (mutually exclusive; cppi_budget_c takes priority if set):
      - plain: cppi=False, cppi_budget_c=None -> size = vol_target_size * leverage.
      - legacy CPPI (cppi=True): session 10's literal Grossman-Zhou-style overlay,
        cushion ratio = cushion / equity -- unchanged, kept exactly as originally
        implemented for backward compatibility with existing callers.
      - budget-normalized CPPI (cppi_budget_c=c): session 11 retune, normalizes the
        cushion against the actual drawdown BUDGET (start_equity - floor) rather than
        against current equity (~1, which made the legacy version's ratio implicitly
        ~33x too conservative regardless of any multiplier). Ramps from 0 at the floor
        to full size once the cushion reaches c * budget, so smaller c means de-risking
        only engages very close to the floor.
    The CPPI ratio is recomputed every bar (genuine dynamic de-risking); the vol-target
    base size is fixed at entry and held through the trade, matching the real signal."""
    block_len = BLOCK_LEN[interval]
    log_ret, high_frac, low_frac = block_bootstrap_path(log_ret_hist, hi_hist, lo_hist, n_bars, block_len, rng)
    close = np.concatenate([[1.0], np.exp(np.cumsum(log_ret))])
    high = close[1:] * np.exp(high_frac)
    low = close[1:] * np.exp(low_frac)

    mom = pd.Series(close).pct_change(lookback_bars).shift(1).to_numpy()
    tr = np.concatenate([[np.nan], np.maximum(high - low, np.maximum(
        np.abs(high - close[:-1]), np.abs(low - close[:-1])))])
    atr = pd.Series(tr).rolling(atr_k).mean().shift(1).to_numpy()
    log_ret_padded = np.concatenate([[np.nan], log_ret])
    realized_vol = pd.Series(log_ret_padded).rolling(VOL_LOOKBACK_BARS).std().shift(1).to_numpy()
    target_vol_per_bar = TARGET_VOL_ANNUAL / np.sqrt(BARS_PER_YEAR[interval])

    bars_per_day = BARS_PER_DAY[interval]
    budget = -STATIC_FLOOR  # start_equity(=1.0) - floor(=1+STATIC_FLOOR)
    equity, day_open, pos_dir, stop_level, pos_vol_scale = 1.0, 1.0, 0, np.nan, 0.0

    def _size(eq: float) -> float:
        base = min(pos_vol_scale, LEVERAGE_CAP[coin]) * leverage
        cushion = max(0.0, eq - (1 + STATIC_FLOOR))
        if cppi_budget_c is not None:
            ratio = min(1.0, cushion / (cppi_budget_c * budget))
        elif cppi:
            ratio = cushion / eq
        else:
            ratio = 1.0
        return min(base * ratio, LEVERAGE_CAP[coin])

    for t in range(1, n_bars + 1):
        prev_close = close[t - 1]
        cur_close = close[t]
        cur_high = high[t - 1]
        cur_low = low[t - 1]

        if (t - 1) % bars_per_day == 0:
            day_open = equity

        raw_ret = 0.0
        if pos_dir == 0:
            if (not np.isnan(mom[t]) and mom[t] != 0 and not np.isnan(atr[t])
                    and not np.isnan(realized_vol[t]) and realized_vol[t] > 0):
                pos_dir = 1 if mom[t] > 0 else -1
                stop_level = prev_close - pos_dir * atr_mult * atr[t]
                pos_vol_scale = target_vol_per_bar / realized_vol[t]
                size = _size(equity)
                raw_ret = size * pos_dir * np.log(cur_close / prev_close) - size * round_trip_cost / 2
        else:
            breached = (cur_low <= stop_level) if pos_dir == 1 else (cur_high >= stop_level)
            size = _size(equity)
            if breached:
                raw_ret = size * pos_dir * np.log(stop_level / prev_close) - size * round_trip_cost / 2
                pos_dir, stop_level, pos_vol_scale = 0, np.nan, 0.0
            else:
                raw_ret = size * pos_dir * np.log(cur_close / prev_close)
                if not np.isnan(atr[t]):
                    stop_level = max(stop_level, cur_close - atr_mult * atr[t]) if pos_dir == 1 else min(stop_level, cur_close + atr_mult * atr[t])

        equity = equity * np.exp(raw_ret)

        if equity <= 1 + STATIC_FLOOR:
            return ("fail_static", t) if return_bars else "fail_static"
        if equity <= day_open * (1 - DAILY_LOSS):
            return ("fail_daily", t) if return_bars else "fail_daily"
        if equity >= 1 + TARGET:
            return ("pass", t) if return_bars else "pass"
    return ("unresolved", n_bars) if return_bars else "unresolved"


def run_barrier_sim(coin: str, interval: str, lookback_bars: int, atr_k: int = 14,
                     atr_mult: float = 2.5, n_paths: int = 500, horizon_days: int = 1095,
                     cppi: bool = False, leverage: float | None = None, seed: int = 0,
                     bar_range: tuple[int, int] | None = None,
                     cppi_budget_c: float | None = None,
                     mia_target_mdd: float = 0.015, mia_horizon_days: int = 1095,
                     round_trip_cost: float = ROUND_TRIP_COST) -> dict:
    """bar_range restricts BOTH the block-bootstrap source data and (when leverage is
    None) the MI&A calibration sample to that bar slice -- e.g. a walk-forward test
    period never seen during lookback-bars/combo selection. horizon_days is the
    simulated deployment length (path length in the Monte Carlo); mia_horizon_days is
    the separate horizon MI&A calibration targets (session 10 conflated these by always
    using 1095 for both -- session 11 retuning needs them independent). round_trip_cost
    defaults to the flat 9bps convention; pass a real per-coin L2-cost figure (session 12,
    research/tsmom_l2_cost_check.py) to override -- also feeds MI&A's own calibration,
    which depends on the same cost-adjusted return series.

    `leverage` is a multiplier k applied on top of genuine per-bar vol-targeted sizing
    (see simulate_path) -- pass leverage=1.0 for the plain validated strategy with no
    overlay (session 14's corrected baseline; NOT session 10-13's old constant-leverage
    stand-in). leverage=None triggers MI&A calibration of k instead of using a fixed one."""
    rng = np.random.default_rng(seed)
    log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval, bar_range)
    if leverage is None:
        leverage, mu, sigma = calibrate_mia_leverage(
            coin, interval, lookback_bars, atr_k, atr_mult, target_mdd=mia_target_mdd,
            bar_range=bar_range, horizon_days=mia_horizon_days, round_trip_cost=round_trip_cost)
    n_bars = int(horizon_days * 24 / int(interval[:-1]))

    outcomes = {"pass": 0, "fail_static": 0, "fail_daily": 0, "unresolved": 0}
    for i in range(n_paths):
        r = simulate_path(coin, interval, lookback_bars, atr_k, atr_mult, leverage, cppi,
                           n_bars, log_ret_hist, hi_hist, lo_hist, rng,
                           cppi_budget_c=cppi_budget_c, round_trip_cost=round_trip_cost)
        outcomes[r] += 1
    return {"leverage": leverage, **{k: v / n_paths for k, v in outcomes.items()}}


if __name__ == "__main__":
    combos = [("BTC", "4h", 1), ("ETH", "12h", 1), ("SOL", "12h", 2)]
    N_PATHS = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    for coin, interval, lb in combos:
        log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval)
        close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret_hist))])
        high_full = np.concatenate([[1.0], close_full[1:] * np.exp(hi_hist)])
        low_full = np.concatenate([[1.0], close_full[1:] * np.exp(lo_hist)])
        _, vt_ret = vol_target_returns(log_ret_hist, high_full, low_full, close_full, lb, 14, 2.5, coin, interval)
        mu, sigma = vt_ret.mean(), vt_ret.std(ddof=1)

        print(f"\n=== {coin} {interval} lookback={lb}: vol-targeted (k=1) mu={mu:.5f}, "
              f"sigma={sigma:.5f} per calendar bar, leverage cap={LEVERAGE_CAP[coin]} ===")

        # Horizon sensitivity: what multiplier k does MI&A imply at a few different
        # calibration horizons, on top of the already-validated k=1 vol-targeted process?
        for days in [14, 30, 90, 180, 365, 1095]:
            bars = int(days * 24 / int(interval[:-1]))
            k = min(calibrate_leverage_for_target_mdd(mu, sigma, bars, 0.015), LEVERAGE_CAP[coin])
            print(f"    T={days:>4}d: MI&A multiplier k for 1.5% E[MDD] = {k:.3f}")

        # Barrier sim: (A) plain k=1 vol-target baseline, (A+CPPI) same + floor de-risking,
        # (B) MI&A @ 1095d (the literal reading of the user's proposal, reported honestly)
        res_a = run_barrier_sim(coin, interval, lb, n_paths=N_PATHS, cppi=False, leverage=1.0)
        print(f"  (A) k=1 vol-target, no overlay:             {res_a}")
        res_a_cppi = run_barrier_sim(coin, interval, lb, n_paths=N_PATHS, cppi=True, leverage=1.0)
        print(f"  (A+CPPI) k=1 + legacy floor de-risking:      {res_a_cppi}")
        k_mia_1095 = min(calibrate_leverage_for_target_mdd(mu, sigma, HORIZON_BARS_1095D[interval], 0.015), LEVERAGE_CAP[coin])
        res_b = run_barrier_sim(coin, interval, lb, n_paths=N_PATHS, cppi=False, leverage=k_mia_1095)
        print(f"  (B) MI&A multiplier @ 1095d horizon, no CPPI: {res_b}")
