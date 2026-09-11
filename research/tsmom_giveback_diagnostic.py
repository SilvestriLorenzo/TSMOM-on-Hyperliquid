"""
H_GIVEBACK: trade-level give-back / tail-concentration diagnostic (session 23,
user-prompted from watching the live Beta charts: the strategy appears to give back a
large slice of unrealized gain in most trades before its ATR stop finally fires, with
total PnL looking carried by a handful of monster-runner trades).

The project's existing outlier-robustness check (H_TSMOM, HYPOTHESES.md session 10) is
BAR-level: drop the 20 largest-magnitude bars, see if aggregate Sharpe survives. It
found TSMOM does NOT have H_ETF's tail-concentration problem at that granularity. That
is a different question from this one: does each individual TRADE give back most of its
own peak unrealized gain before it closes, and is total realized PnL itself concentrated
in a small number of trades? Nothing in the repo has measured this before -- this is a
new, standalone diagnostic on the already-validated backtest, not a new strategy variant.

Method: duplicate `tsmom_walkforward._strat_ret_series`'s exact entry/ATR-stop loop (same
duplicate-rather-than-touch-validated-code convention every other script here uses), but
instead of only accumulating the per-bar strat_ret series, emit one row per closed trade:
entry/exit time & price, direction, size (vol-target size fixed at entry, matching
production), realized trade return, and MFE (max favorable excursion) -- the best
intrabar price available to the trade at any point between entry and exit, tracked
causally bar-by-bar using the same intrabar high/low convention the ATR stop itself
checks breaches against.

IMPORTANT: computing MFE over a trade's own already-realized bars is NOT a look-ahead
bug in the sense H_TSMOM's session-10 bug was (checking a stop against information not
yet available to a live decision). MFE here is a purely retrospective property of a
CLOSED trade, computed after the fact for diagnostic purposes -- it is never fed back
into a live entry/exit decision. The distinction matters enough to state explicitly given
this project's history with exactly this failure mode.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m
from tsmom_walkforward import walk_forward_one
from tsmom_joint_portfolio_barrier_sim import LOCKED

OUT = Path(__file__).parent / "output"
ATR_K = 14
ATR_MULT = 2.5


def extract_trades(coin: str, interval: str, lookback_bars: int,
                    atr_k: int = ATR_K, atr_mult: float = ATR_MULT,
                    return_bar_series: bool = False):
    """Duplicates backtest_trailing_stop's/_strat_ret_series' exact loop bar-for-bar
    (same pos_size/stop_level/entry-condition logic, same cost convention: half the
    round-trip cost charged on the entry bar, half on the exit bar), emitting one row
    per closed trade. A trade still open at the end of history is dropped (no realized
    exit to measure). `realized_ret` per trade is the trade's own log-return, telescoped
    from entry_px to exit_px (valid because pos_size/pos_dir are constant for the
    trade's duration, so the per-bar log-return components sum exactly to
    log(exit_px/entry_px) regardless of the intermediate path) plus the round-trip cost.

    If return_bar_series=True, also returns the full per-bar strat_ret array/index
    exactly as backtest_trailing_stop produces it (used only for the reconciliation
    sanity check below, not for the give-back analysis itself, which uses per-trade
    realized_ret)."""
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
    strat_ret = np.zeros(n)  # full per-bar series, same as backtest_trailing_stop's own
    pos_size, pos_dir, stop_level = 0.0, 0, np.nan
    entry_idx, entry_px = -1, np.nan
    mfe_px = np.nan  # best intrabar price seen so far this trade, in the FAVORABLE direction

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if (not np.isnan(mom_a[t]) and mom_a[t] != 0 and not np.isnan(atr_a[t])
                    and not np.isnan(vol_a[t]) and vol_a[t] > 0):
                pos_dir = 1 if mom_a[t] > 0 else -1
                pos_size = min(target_vol_per_bar / vol_a[t], m.LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr_a[t]
                entry_idx, entry_px = t, prev_close
                mfe_px = high[t] if pos_dir == 1 else low[t]
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * cost / 2
        else:
            mfe_px = max(mfe_px, high[t]) if pos_dir == 1 else min(mfe_px, low[t])
            breached = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            if breached:
                exit_px = stop_level
                strat_ret[t] = pos_dir * pos_size * np.log(exit_px / prev_close) - pos_size * cost / 2
                realized_ret = pos_dir * pos_size * np.log(exit_px / entry_px) - pos_size * cost
                mfe_ret = pos_dir * pos_size * np.log(mfe_px / entry_px)
                trades.append({
                    "coin": coin, "interval": interval, "entry_idx": entry_idx,
                    "entry_time": times[entry_idx], "entry_px": entry_px,
                    "exit_idx": t, "exit_time": times[t], "exit_px": exit_px,
                    "pos_dir": pos_dir, "pos_size": pos_size,
                    "realized_ret": realized_ret, "mfe_ret": mfe_ret,
                })
                pos_dir, pos_size, stop_level = 0, 0.0, np.nan
                entry_idx, entry_px, mfe_px = -1, np.nan, np.nan
            else:
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr_a[t]):
                    if pos_dir == 1:
                        stop_level = max(stop_level, close[t] - atr_mult * atr_a[t])
                    else:
                        stop_level = min(stop_level, close[t] + atr_mult * atr_a[t])

    trades_df = pd.DataFrame(trades)
    if return_bar_series:
        return trades_df, pd.Series(strat_ret, index=df.index).iloc[1:]
    return trades_df


def run_all_combos() -> pd.DataFrame:
    frames = []
    for (coin, interval), (lb, _k) in LOCKED.items():
        t = extract_trades(coin, interval, lb)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _giveback_col(trades: pd.DataFrame) -> pd.Series:
    """giveback_ratio = (mfe - realized) / mfe, defined only where mfe > 0 (a trade that
    was never favorable has no peak gain to give back -- bucketed separately, not
    forced to 0 or 1)."""
    favorable = trades["mfe_ret"] > 0
    ratio = pd.Series(np.nan, index=trades.index)
    ratio[favorable] = (trades.loc[favorable, "mfe_ret"] - trades.loc[favorable, "realized_ret"]) / trades.loc[favorable, "mfe_ret"]
    return ratio


def giveback_summary(trades: pd.DataFrame) -> dict:
    gb = _giveback_col(trades)
    favorable = trades["mfe_ret"] > 0
    n = len(trades)
    gb_fav = gb[favorable]
    return {
        "n_trades": n,
        "n_favorable": int(favorable.sum()),
        "pct_never_favorable": 1 - favorable.mean() if n else np.nan,
        "gb_mean": gb_fav.mean(), "gb_median": gb_fav.median(),
        "gb_p25": gb_fav.quantile(0.25), "gb_p75": gb_fav.quantile(0.75), "gb_p90": gb_fav.quantile(0.90),
        "pct_gb_gt_50": (gb_fav > 0.5).mean() if len(gb_fav) else np.nan,
        "pct_gb_gt_80": (gb_fav > 0.8).mean() if len(gb_fav) else np.nan,
    }


def gini(x: pd.Series) -> float:
    """Standard Gini coefficient, computed on realized_ret directly (can include
    negative values -- interpreted here as a concentration measure among the
    positive-contribution trades' share of total, see pnl_concentration)."""
    v = np.sort(x.to_numpy())
    n = len(v)
    if n == 0 or v.sum() == 0:
        return np.nan
    cum = np.cumsum(v)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n if cum[-1] != 0 else np.nan


def pnl_concentration(trades: pd.DataFrame) -> dict:
    """Among WINNING trades only (realized_ret > 0): what fraction of total winning PnL
    is contributed by the top 5/10/20% largest winners, plus a Gini coefficient on the
    winning-trade return distribution. This is the trade-level analogue of H_TSMOM's
    existing bar-level "drop the 20 largest bars" outlier check."""
    wins = trades.loc[trades["realized_ret"] > 0, "realized_ret"].sort_values(ascending=False)
    total = wins.sum()
    n = len(wins)
    out = {"n_winning_trades": n, "total_winning_pnl": total}
    for pct in (0.05, 0.10, 0.20):
        k = max(1, int(np.ceil(n * pct)))
        out[f"top_{int(pct*100)}pct_share"] = wins.iloc[:k].sum() / total if total > 0 else np.nan
    out["gini_winning_trades"] = gini(wins)
    return out


def drop_top_n(trades: pd.DataFrame, n_drop: int) -> pd.DataFrame:
    """Outlier-robustness check: drop the n_drop largest-magnitude trades (by
    |realized_ret|), matching the project's bar-level drop-N convention applied here at
    trade level."""
    keep = trades["realized_ret"].abs().sort_values(ascending=False).index[n_drop:]
    return trades.loc[keep]


def _reconcile(coin: str, interval: str, lookback_bars: int, bar_series: pd.Series) -> tuple[float, float]:
    """Sanity check: replaying extract_trades' own per-bar strat_ret array through the
    SAME (1+x).cumprod() equity convention backtest_trailing_stop uses (strat_ret is a
    log-return-scaled quantity but is compounded there as if a simple return -- an
    existing convention in the validated code, not something to "fix" here, just
    replicate exactly) must reproduce its reported total_return. This checks that
    extract_trades' loop is bar-for-bar identical to backtest_trailing_stop's, not that
    the give-back metric's own (correct, telescoped-log) realized_ret matches a
    differently-conventioned number."""
    bt = m.backtest_trailing_stop(coin, interval, lookback_bars)
    equity = (1 + bar_series).cumprod()
    return equity.iloc[-1] - 1, bt["total_return"]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    all_trades = run_all_combos()

    print("=== Reconciliation check (replayed bar series must match backtest_trailing_stop's total_return) ===")
    ok = True
    for (coin, interval), (lb, _k) in LOCKED.items():
        _, bar_series = extract_trades(coin, interval, lb, return_bar_series=True)
        implied, reported = _reconcile(coin, interval, lb, bar_series)
        match = np.isclose(implied, reported, atol=1e-6)
        ok &= match
        print(f"  {coin} {interval}: replayed={implied:.4%} vs backtest_trailing_stop={reported:.4%} "
              f"{'OK' if match else 'MISMATCH -- DO NOT TRUST DOWNSTREAM NUMBERS'}")
    if not ok:
        sys.exit("Reconciliation failed -- fix extract_trades before trusting anything below.")

    print("\n=== Give-back summary, per combo and pooled ===")
    rows = []
    for (coin, interval), (lb, _k) in LOCKED.items():
        t = all_trades[(all_trades["coin"] == coin) & (all_trades["interval"] == interval)]
        rows.append({"coin": coin, "interval": interval, **giveback_summary(t)})
    rows.append({"coin": "POOLED", "interval": "", **giveback_summary(all_trades)})
    df = pd.DataFrame(rows)
    fmt = {c: "{:.1%}".format for c in df.columns if c.startswith("gb_") or c.startswith("pct_")}
    print(df.to_string(index=False, formatters=fmt))

    print("\n=== PnL concentration (winning trades only), per combo and pooled ===")
    rows2 = []
    for (coin, interval), (lb, _k) in LOCKED.items():
        t = all_trades[(all_trades["coin"] == coin) & (all_trades["interval"] == interval)]
        rows2.append({"coin": coin, "interval": interval, **pnl_concentration(t)})
    rows2.append({"coin": "POOLED", "interval": "", **pnl_concentration(all_trades)})
    df2 = pd.DataFrame(rows2)
    fmt2 = {c: "{:.1%}".format for c in df2.columns if "share" in c or c == "gini_winning_trades"}
    fmt2["total_winning_pnl"] = "{:.2%}".format
    print(df2.to_string(index=False, formatters=fmt2))

    print("\n=== Outlier-robustness: give-back & concentration after dropping top-5 / top-10 largest trades (pooled) ===")
    for n_drop in (5, 10):
        trimmed = drop_top_n(all_trades, n_drop)
        gb = giveback_summary(trimmed)
        pc = pnl_concentration(trimmed)
        print(f"  drop top {n_drop}: gb_median={gb['gb_median']:.1%}, pct_gb_gt_50={gb['pct_gb_gt_50']:.1%}, "
              f"top10pct_share={pc['top_10pct_share']:.1%}, gini={pc['gini_winning_trades']:.2f}")

    print("\n=== Train/test temporal stability (walk_forward_one's own bar_range split, per combo) ===")
    for (coin, interval), (lb, _k) in LOCKED.items():
        wf = walk_forward_one(coin, interval)
        split, end = wf["bar_range"]
        t = all_trades[(all_trades["coin"] == coin) & (all_trades["interval"] == interval)]
        train = t[t["entry_idx"] < split]
        test = t[t["entry_idx"] >= split]
        gb_train, gb_test = giveback_summary(train), giveback_summary(test)
        print(f"  {coin} {interval}: train gb_median={gb_train['gb_median']:.1%} (n={gb_train['n_trades']}) | "
              f"test gb_median={gb_test['gb_median']:.1%} (n={gb_test['n_trades']})")

    all_trades.to_json(OUT / "tsmom_giveback_trades.json", orient="records", indent=2, date_format="iso")
    print(f"\nWrote {len(all_trades)} trade records to {OUT / 'tsmom_giveback_trades.json'}")
