"""
H_PARTIAL_SCALEOUT: bank part of a trade's unrealized gain without cutting the runner
(session 28, user-proposed reframe: this thread is no longer optimizing P(pass) --
this project isn't the user's main income and every dollar risked is treated as
"tuition" -- the actual goal is a more normalized return distribution (win rate is
below 50%, edge is carried by a minority of monster winners per H_GIVEBACK) and, if
possible, FASTER strategy resolution (explicit user framing: "a strategy with 50%
p_pass that resolves in 10 days is useful; one with 50% p_pass that resolves in 40
days provides no advantage" -- speed matters as much as the pass fraction itself).

H_GIVEBACK (session 23) quantified exactly the pattern motivating this: pooled median
give-back ratio 119% (the median trade that is ever green still closes a net LOSER by
the time its ATR stop fires), ~43% win rate, winning-trade PnL meaningfully but not
extremely concentrated (Gini 0.58, top 10% of winners = 44% of winning PnL). Every
mechanism tried since (H_TP_EXIT, H_POST_TP_REENTRY, H_POST_TP_SCORE_V2) used a FULL
exit -- either a capped take-profit replacing/racing the stop, or a full exit followed
by an independent re-entry/flip/flat decision. H_TP_EXIT's capped TP was a clean kill
specifically because it cuts the runner. Nothing in this repo has tested a PARTIAL
scale-out: bank a fraction of size once a trade reaches a give-back-informed favorable
excursion, and leave the REMAINDER running on the exact same, completely unmodified
ATR-2.5x trailing stop -- the runner itself is never touched, only downsized once.

Method: `vol_target_returns_partial_scaleout` is a parallel copy of
`tsmom_barrier_sim.vol_target_returns` (same duplicate-rather-than-touch-validated-code
convention as every variant in this project) that adds a single scale-out event per
trade: a trigger level fixed at entry (entry_px +/- trigger_atr_mult * ATR at entry,
never ratchets -- same "fixed at entry" convention as `tsmom_exit_variants`' TP level),
checked against bar t's own intrabar high/low (same causal-timing discipline as every
stop/TP check here: level fixed using data through t-1, breach checked against bar t).
When first touched, `bank_frac` of the CURRENT size is realized at the trigger price
(paying round-trip cost on the banked notional only); the remainder continues at
reduced size on the SAME stop_level, which keeps ratcheting exactly as before. Fires at
most once per trade; bank_frac=0.0 is a no-op and must reproduce
`vol_target_returns` bar-for-bar (checked below).

Grid search over (trigger_atr_mult x bank_frac), reusing `tsmom_exit_variants.py`'s
generic walk-forward/barrier-sim/outlier-check machinery unmodified (it was written
generically over any variant_fn matching vol_target_returns' calling convention).
Candidate SELECTION still uses the project's standard train-only Sharpe discipline
(not one of the new distribution-shape metrics directly) to avoid a fresh overfitting
risk on metrics that are easier to game with a small trade count than Sharpe is; the
full grid's held-out TEST-slice numbers -- across Sharpe, P(pass), median days to
resolution, win rate, give-back, skew/kurtosis, and winning-trade Gini -- are reported
transparently for every candidate, not just the selected one, since this round is
explicitly exploratory (the user wants to see the tradeoff surface, not just one
locked pick).

Scope: research only, same as every other thread in this project. Does not touch
tsmom_beta_live.py, tsmom_paper_trader.py, or the locked deployment config. Uses the
same on-disk candle data as every other post-session-26 script here (no new fetch), so
it does not touch bars after the session-26 holdout cutoff (research/output/
tsmom_holdout_cutoff.json) -- that lock is specifically about H_POST_TP_SCORE_V2's own
frozen confirmation run, not a bar on other threads using the existing data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m
from tsmom_barrier_sim import extract_market_tuples, block_bootstrap_path, vol_target_returns, TARGET, STATIC_FLOOR, DAILY_LOSS, BLOCK_LEN
from tsmom_joint_portfolio_barrier_sim import LOCKED
from tsmom_giveback_diagnostic import extract_trades as extract_trades_baseline, gini
from tsmom_exit_variants import (
    walk_forward_variant, _real_history_series, _sharpe, ATR_K, ATR_MULT_BASE, BARS_PER_DAY, HORIZON_DAYS,
)

OUT = Path(__file__).parent / "output"
N_PATHS = 250  # reduced from tsmom_exit_variants' default 400: this grid (13 params/combo) is
               # ~2.3x larger, keeping total runtime comparable
TRIGGER_ATR_GRID = [0.75, 1.0, 1.5, 2.0]
BANK_FRAC_GRID = [0.25, 0.5, 0.75]
BASELINE_PARAM = {"atr_mult": ATR_MULT_BASE, "trigger_atr_mult": TRIGGER_ATR_GRID[0], "bank_frac": 0.0}
GRID = [BASELINE_PARAM] + [
    {"atr_mult": ATR_MULT_BASE, "trigger_atr_mult": tg, "bank_frac": bf}
    for tg in TRIGGER_ATR_GRID for bf in BANK_FRAC_GRID
]


def vol_target_returns_partial_scaleout(log_ret, high, low, close, lookback_bars, atr_k,
                                          atr_mult, trigger_atr_mult, bank_frac, coin, interval,
                                          round_trip_cost=m.ROUND_TRIP_COST):
    """Same entry/sizing/stop as vol_target_returns. Once favorable excursion first
    touches trigger_level (fixed at entry, never ratchets), bank_frac of the CURRENT
    size is realized there (round-trip cost charged on the banked notional only) and
    the remainder keeps running on the SAME, unmodified stop_level -- fires at most
    once per trade. bank_frac=0.0 disables the mechanism entirely (no-op)."""
    n = len(close)
    mom = pd.Series(close).pct_change(lookback_bars).shift(1).to_numpy()
    tr = np.concatenate([[np.nan], np.maximum(high[1:] - low[1:], np.maximum(
        np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))])
    atr = pd.Series(tr).rolling(atr_k).mean().shift(1).to_numpy()
    log_ret_padded = np.concatenate([[np.nan], log_ret])
    realized_vol = pd.Series(log_ret_padded).rolling(m.VOL_LOOKBACK_BARS).std().shift(1).to_numpy()
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(m.BARS_PER_YEAR[interval])

    pos_dir_arr = np.zeros(n)
    strat_ret = np.zeros(n)
    pos_dir, stop_level, trigger_level, pos_size, scaled_out = 0, np.nan, np.nan, 0.0, False

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if (not np.isnan(mom[t]) and mom[t] != 0 and not np.isnan(atr[t])
                    and not np.isnan(realized_vol[t]) and realized_vol[t] > 0):
                pos_dir = 1 if mom[t] > 0 else -1
                pos_size = min(target_vol_per_bar / realized_vol[t], m.LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr[t]
                trigger_level = prev_close + pos_dir * trigger_atr_mult * atr[t]
                scaled_out = False
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * round_trip_cost / 2
        else:
            breached = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            if breached:
                strat_ret[t] = pos_dir * pos_size * np.log(stop_level / prev_close) - pos_size * round_trip_cost / 2
                pos_dir, stop_level, trigger_level, pos_size, scaled_out = 0, np.nan, np.nan, 0.0, False
            else:
                trig_hit = (not scaled_out) and bank_frac > 0 and (
                    (high[t] >= trigger_level) if pos_dir == 1 else (low[t] <= trigger_level))
                if trig_hit:
                    banked_size = pos_size * bank_frac
                    remainder_size = pos_size - banked_size
                    banked_leg = pos_dir * banked_size * np.log(trigger_level / prev_close) - banked_size * round_trip_cost / 2
                    remainder_leg = pos_dir * remainder_size * np.log(close[t] / prev_close)
                    strat_ret[t] = banked_leg + remainder_leg
                    pos_size = remainder_size
                    scaled_out = True
                else:
                    strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr[t]):
                    if pos_dir == 1:
                        stop_level = max(stop_level, close[t] - atr_mult * atr[t])
                    else:
                        stop_level = min(stop_level, close[t] + atr_mult * atr[t])
        pos_dir_arr[t] = pos_dir
    return pos_dir_arr, strat_ret


def extract_trades_partial(coin, interval, lookback_bars, trigger_atr_mult, bank_frac,
                             atr_k=ATR_K, atr_mult=ATR_MULT_BASE):
    """Duplicates vol_target_returns_partial_scaleout's exact loop (same convention as
    tsmom_giveback_diagnostic.extract_trades) to emit one row per closed trade.
    `realized_ret` is the SUM of the trade's own bar-level strat_ret contributions --
    correct across a scale-out event since each bar's contribution already reflects
    whatever size was active and whatever cost was charged that bar -- a size-weighted
    BLEND of the banked leg and the remainder leg, directly comparable to H_GIVEBACK's
    realized_ret for a same-size (never-scaled) trade. `mfe_ret` uses the trade's
    INITIAL (pre-scale-out) size throughout, matching H_GIVEBACK's own definition
    exactly, so giveback_ratio measures "of the paper profit the ORIGINAL full-size
    position could have captured, how much did this mechanism actually keep" -- the
    literal question this hypothesis asks."""
    cost = m.ROUND_TRIP_COST
    df = m.load_candles(coin, interval)
    log_ret = np.log1p(df["c"].pct_change())
    mom = df["c"].pct_change(lookback_bars).shift(1)
    tr = np.maximum(df["h"] - df["l"], np.maximum((df["h"] - df["c"].shift(1)).abs(),
                                                    (df["l"] - df["c"].shift(1)).abs()))
    atr = tr.rolling(atr_k).mean().shift(1)
    realized_vol = log_ret.rolling(m.VOL_LOOKBACK_BARS).std().shift(1)
    bars_per_year = m.BARS_PER_YEAR[interval]
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(bars_per_year)

    close, high, low = df["c"].to_numpy(), df["h"].to_numpy(), df["l"].to_numpy()
    times = df.index.to_numpy()
    mom_a, atr_a, vol_a = mom.to_numpy(), atr.to_numpy(), realized_vol.to_numpy()
    n = len(df)

    trades = []
    pos_size, pos_dir, stop_level, trigger_level, scaled_out = 0.0, 0, np.nan, np.nan, False
    entry_idx, entry_px, init_size = -1, np.nan, 0.0
    mfe_px = np.nan
    trade_ret_accum, banked_leg_ret = 0.0, 0.0

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if (not np.isnan(mom_a[t]) and mom_a[t] != 0 and not np.isnan(atr_a[t])
                    and not np.isnan(vol_a[t]) and vol_a[t] > 0):
                pos_dir = 1 if mom_a[t] > 0 else -1
                pos_size = min(target_vol_per_bar / vol_a[t], m.LEVERAGE_CAP[coin])
                init_size = pos_size
                stop_level = prev_close - pos_dir * atr_mult * atr_a[t]
                trigger_level = prev_close + pos_dir * trigger_atr_mult * atr_a[t]
                scaled_out = False
                entry_idx, entry_px = t, prev_close
                mfe_px = high[t] if pos_dir == 1 else low[t]
                trade_ret_accum = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * cost / 2
                banked_leg_ret = 0.0
        else:
            mfe_px = max(mfe_px, high[t]) if pos_dir == 1 else min(mfe_px, low[t])
            breached = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            if breached:
                exit_px = stop_level
                trade_ret_accum += pos_dir * pos_size * np.log(exit_px / prev_close) - pos_size * cost / 2
                mfe_ret = pos_dir * init_size * np.log(mfe_px / entry_px)
                trades.append({
                    "coin": coin, "interval": interval, "entry_idx": entry_idx,
                    "entry_time": times[entry_idx], "exit_idx": t, "exit_time": times[t],
                    "pos_dir": pos_dir, "init_size": init_size, "scaled_out": scaled_out,
                    "realized_ret": trade_ret_accum, "banked_leg_ret": banked_leg_ret,
                    "mfe_ret": mfe_ret,
                })
                pos_dir, pos_size, stop_level, trigger_level, scaled_out = 0, 0.0, np.nan, np.nan, False
                entry_idx, entry_px, init_size, mfe_px = -1, np.nan, 0.0, np.nan
                trade_ret_accum, banked_leg_ret = 0.0, 0.0
            else:
                trig_hit = (not scaled_out) and bank_frac > 0 and (
                    (high[t] >= trigger_level) if pos_dir == 1 else (low[t] <= trigger_level))
                if trig_hit:
                    banked_size = pos_size * bank_frac
                    remainder_size = pos_size - banked_size
                    banked_leg = pos_dir * banked_size * np.log(trigger_level / prev_close) - banked_size * cost / 2
                    remainder_leg = pos_dir * remainder_size * np.log(close[t] / prev_close)
                    trade_ret_accum += banked_leg + remainder_leg
                    banked_leg_ret = banked_leg
                    pos_size = remainder_size
                    scaled_out = True
                else:
                    trade_ret_accum += pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr_a[t]):
                    if pos_dir == 1:
                        stop_level = max(stop_level, close[t] - atr_mult * atr_a[t])
                    else:
                        stop_level = min(stop_level, close[t] + atr_mult * atr_a[t])
    return pd.DataFrame(trades)


def simulate_path_days(coin, interval, lookback_bars, params, n_bars, log_ret_hist, hi_hist, lo_hist,
                         rng, round_trip_cost=m.ROUND_TRIP_COST):
    """Parallel to tsmom_exit_variants.simulate_path_variant, additionally returning
    bars-to-resolution -- needed to test whether this mechanism resolves FASTER, not
    just at a different pass rate (the user's explicit "does it accelerate the
    strategy" framing: a variant with the same p_pass that resolves sooner is strictly
    better, one that resolves later is not, even at equal p_pass)."""
    block_len = BLOCK_LEN[interval]
    log_ret, high_frac, low_frac = block_bootstrap_path(log_ret_hist, hi_hist, lo_hist, n_bars, block_len, rng)
    close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret))])
    high_full = np.concatenate([[1.0], close_full[1:] * np.exp(high_frac)])
    low_full = np.concatenate([[1.0], close_full[1:] * np.exp(low_frac)])

    _, strat_ret = vol_target_returns_partial_scaleout(
        log_ret, high_full, low_full, close_full, lookback_bars, ATR_K,
        coin=coin, interval=interval, round_trip_cost=round_trip_cost, **params)

    bars_per_day = BARS_PER_DAY[interval]
    equity, day_open = 1.0, 1.0
    for t in range(1, len(close_full)):
        if (t - 1) % bars_per_day == 0:
            day_open = equity
        equity *= np.exp(strat_ret[t])
        if equity <= 1 + STATIC_FLOOR:
            return "fail_static", t
        if equity <= day_open * (1 - DAILY_LOSS):
            return "fail_daily", t
        if equity >= 1 + TARGET:
            return "pass", t
    return "unresolved", len(close_full) - 1


def run_barrier_sim_days(coin, interval, lookback_bars, params, bar_range=None,
                           n_paths=N_PATHS, horizon_days=HORIZON_DAYS, seed=0):
    rng = np.random.default_rng(seed)
    log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval, bar_range)
    n_bars = int(horizon_days * 24 / int(interval[:-1]))
    bars_per_day = BARS_PER_DAY[interval]
    outcomes = {"pass": 0, "fail_static": 0, "fail_daily": 0, "unresolved": 0}
    days_all, days_pass = [], []
    for _ in range(n_paths):
        r, bars_to_res = simulate_path_days(coin, interval, lookback_bars, params, n_bars,
                                              log_ret_hist, hi_hist, lo_hist, rng)
        outcomes[r] += 1
        if r != "unresolved":
            days_all.append(bars_to_res / bars_per_day)
            if r == "pass":
                days_pass.append(bars_to_res / bars_per_day)
    return {
        **{k: v / n_paths for k, v in outcomes.items()},
        "median_days": float(np.median(days_all)) if days_all else float("nan"),
        "median_days_pass": float(np.median(days_pass)) if days_pass else float("nan"),
    }


def _gb_median(trades: pd.DataFrame) -> float:
    fav = trades[trades["mfe_ret"] > 0]
    if len(fav) == 0:
        return float("nan")
    return ((fav["mfe_ret"] - fav["realized_ret"]) / fav["mfe_ret"]).median()


def _param_label(p: dict) -> str:
    return "baseline (no scale-out)" if p["bank_frac"] == 0.0 else f"trig={p['trigger_atr_mult']}xATR, bank={p['bank_frac']:.0%}"


if __name__ == "__main__":
    print("=== Sanity check 1: bank_frac=0.0 reproduces vol_target_returns bar-for-bar ===")
    for (coin, interval), (lb, _k) in LOCKED.items():
        log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval)
        close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret_hist))])
        high_full = np.concatenate([[1.0], close_full[1:] * np.exp(hi_hist)])
        low_full = np.concatenate([[1.0], close_full[1:] * np.exp(lo_hist)])
        _, ref = vol_target_returns(log_ret_hist, high_full, low_full, close_full, lb, ATR_K, ATR_MULT_BASE, coin, interval)
        _, test = vol_target_returns_partial_scaleout(log_ret_hist, high_full, low_full, close_full, lb, ATR_K,
                                                        **BASELINE_PARAM, coin=coin, interval=interval)
        match = np.allclose(ref, test, atol=1e-10)
        print(f"  {coin} {interval}: {'OK' if match else 'MISMATCH'}")
        if not match:
            sys.exit("bank_frac=0 baseline-reproduction sanity check failed.")

    print("\n=== Sanity check 2: extract_trades_partial(bank_frac=0.0) matches tsmom_giveback_diagnostic.extract_trades ===")
    for (coin, interval), (lb, _k) in LOCKED.items():
        base = extract_trades_baseline(coin, interval, lb)
        part = extract_trades_partial(coin, interval, lb, trigger_atr_mult=BASELINE_PARAM["trigger_atr_mult"], bank_frac=0.0)
        match = (len(base) == len(part)
                 and np.allclose(base["realized_ret"].to_numpy(), part["realized_ret"].to_numpy(), atol=1e-10)
                 and np.allclose(base["mfe_ret"].to_numpy(), part["mfe_ret"].to_numpy(), atol=1e-10))
        print(f"  {coin} {interval}: {'OK' if match else 'MISMATCH'} (n_base={len(base)}, n_part={len(part)})")
        if not match:
            sys.exit("extract_trades_partial baseline-reproduction sanity check failed.")

    print(f"\n=== Grid: trigger_atr_mult in {TRIGGER_ATR_GRID}, bank_frac in {BANK_FRAC_GRID} "
          f"(+ baseline), {N_PATHS} MC paths/candidate ===")

    per_combo_rows = []
    pooled_trades_by_param = {i: [] for i in range(len(GRID))}

    for (coin, interval), (lb, _k) in LOCKED.items():
        wf = walk_forward_variant(coin, interval, lb, vol_target_returns_partial_scaleout, GRID)
        split, end = wf["bar_range"]
        star_label = _param_label(wf["param_star"])
        print(f"\n--- {coin} {interval}: train-Sharpe-selected candidate = {star_label} "
              f"(train Sharpe {wf['train_sharpe_at_star']:.2f}, test Sharpe {wf['test_sharpe']:.2f}) ---")

        for i, params in enumerate(GRID):
            s = _real_history_series(coin, interval, lb, vol_target_returns_partial_scaleout, **params)
            test_sharpe = _sharpe(s.iloc[split:], m.BARS_PER_YEAR[interval])
            sim = run_barrier_sim_days(coin, interval, lb, params, bar_range=(split, end))
            trades = extract_trades_partial(coin, interval, lb, trigger_atr_mult=params["trigger_atr_mult"],
                                              bank_frac=params["bank_frac"], atr_mult=params["atr_mult"])
            test_trades = trades[trades["entry_idx"] >= split]
            pooled_trades_by_param[i].append(test_trades)

            is_star = params == wf["param_star"]
            per_combo_rows.append({
                "coin": coin, "interval": interval, "param": _param_label(params),
                "selected": "*" if is_star else "", "test_sharpe": test_sharpe,
                "p_pass": sim["pass"], "median_days": sim["median_days"],
                "median_days_pass": sim["median_days_pass"], "n_trades": len(test_trades),
                "win_rate": (test_trades["realized_ret"] > 0).mean() if len(test_trades) else np.nan,
                "gb_median": _gb_median(test_trades),
            })

    per_combo_df = pd.DataFrame(per_combo_rows)
    fmt = {"test_sharpe": "{:.2f}".format, "p_pass": "{:.1%}".format,
           "median_days": "{:.0f}".format, "median_days_pass": "{:.0f}".format,
           "win_rate": "{:.1%}".format, "gb_median": "{:.1%}".format}
    print("\n=== Per-combo, full grid (test/OOS slice only; '*' = train-Sharpe-selected candidate) ===")
    print(per_combo_df.to_string(index=False, formatters=fmt))

    print("\n=== Pooled trade-level distribution shape, by candidate (test/OOS trades, all 4 combos) ===")
    pooled_rows = []
    for i, params in enumerate(GRID):
        pooled = pd.concat(pooled_trades_by_param[i], ignore_index=True)
        wins = pooled.loc[pooled["realized_ret"] > 0, "realized_ret"]
        pooled_rows.append({
            "param": _param_label(params), "n_trades": len(pooled),
            "win_rate": (pooled["realized_ret"] > 0).mean(),
            "gb_median": _gb_median(pooled),
            "skew": pooled["realized_ret"].skew(), "kurt": pooled["realized_ret"].kurt(),
            "gini_winners": gini(wins), "pct_scaled_out": pooled["scaled_out"].mean(),
        })
    pooled_df = pd.DataFrame(pooled_rows)
    fmt2 = {"win_rate": "{:.1%}".format, "gb_median": "{:.1%}".format, "skew": "{:.2f}".format,
            "kurt": "{:.2f}".format, "gini_winners": "{:.2f}".format, "pct_scaled_out": "{:.1%}".format}
    print(pooled_df.to_string(index=False, formatters=fmt2))

    OUT.mkdir(exist_ok=True)
    per_combo_df.to_json(OUT / "tsmom_partial_scaleout_per_combo.json", orient="records", indent=2)
    pooled_df.to_json(OUT / "tsmom_partial_scaleout_pooled.json", orient="records", indent=2)
    print(f"\nSaved to {OUT / 'tsmom_partial_scaleout_per_combo.json'} and "
          f"{OUT / 'tsmom_partial_scaleout_pooled.json'}")
