"""
H_TSMOM walk-forward validation (session 11). Session 10's headline numbers (BTC 4h
lb=1, ETH 12h lb=1, SOL 12h lb=2, and every other combo's Sharpe/p_pass) all picked the
best lookback_bars, and implicitly the best coin/interval to headline, using the exact
same full-history window the numbers were then reported on -- flagged explicitly as an
open gap, unlike H_ETF which has a genuine walk-forward. This closes that gap.

Method: for each of the 9 (coin, interval) combos, split the full candle history
chronologically 70% train / 30% test (no shuffling -- this is a time series). Select
lookback_bars in {1,2,3} by Sharpe on TRAIN ONLY, then report performance on TEST ONLY
under that selected lookback -- genuinely out-of-sample, both for the parameter choice
and (since all 9 combos get this treatment, not just the original top 3) for which
coin/interval looks best. Also reports what test-period Sharpe would have been under the
test-optimal lookback, to quantify how much of the original edge was selection artifact.

The out-of-sample p_pass reuses tsmom_barrier_sim.run_barrier_sim with bar_range
restricted to the test slice and leverage=1.0 (session 14: the barrier sim now runs
genuine per-bar vol-targeted sizing internally -- k=1.0 reproduces the plain validated
strategy with no MI&A/CPPI overlay, no separately-computed constant needed), so the
Monte Carlo's block-bootstrap source draws only from data never used to pick lb*.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_hypothesis import BARS_PER_YEAR
from tsmom_barrier_sim import run_barrier_sim

OUT = Path(__file__).parent / "output"
TRAIN_FRAC = 0.7
COMBOS = [(coin, interval) for coin in ["BTC", "ETH", "SOL"] for interval in ["4h", "8h", "12h"]]
LOOKBACKS = [1, 2, 3]


def sharpe(strat_ret: pd.Series, bars_per_year: int) -> float:
    if len(strat_ret) < 2 or strat_ret.std() == 0:
        return np.nan
    return (strat_ret.mean() / strat_ret.std()) * np.sqrt(bars_per_year)


def walk_forward_one(coin: str, interval: str, round_trip_cost: float | None = None) -> dict:
    """round_trip_cost overrides the flat 9bps convention (None = use
    tsmom_hypothesis.ROUND_TRIP_COST) -- pass a real per-coin L2-cost figure (session 12,
    research/tsmom_l2_cost_check.py) to validate the walk-forward result against it."""
    bars_per_year = BARS_PER_YEAR[interval]

    # backtest_trailing_stop's rolling stats (realized vol, ATR) are already causal
    # bar-by-bar, so slicing its per-bar return series by date does not leak
    # test-period information into train-period decisions -- the only thing selected
    # on train is which lb's TRAIN-slice Sharpe is highest. backtest_trailing_stop
    # itself only returns summary stats, so _strat_ret_series duplicates its exact
    # loop to get the per-bar series needed for slicing.
    series_by_lb = {lb: _strat_ret_series(coin, interval, lb, round_trip_cost) for lb in LOOKBACKS}

    n_bars_series = len(next(iter(series_by_lb.values())))
    split = int(n_bars_series * TRAIN_FRAC)

    train_sharpe = {lb: sharpe(s.iloc[:split], bars_per_year) for lb, s in series_by_lb.items()}
    lb_star = max(train_sharpe, key=lambda k: (train_sharpe[k] if not np.isnan(train_sharpe[k]) else -np.inf))

    test_sharpe_at_lbstar = sharpe(series_by_lb[lb_star].iloc[split:], bars_per_year)
    test_sharpe_by_lb = {lb: sharpe(s.iloc[split:], bars_per_year) for lb, s in series_by_lb.items()}
    lb_test_optimal = max(test_sharpe_by_lb, key=lambda k: (test_sharpe_by_lb[k] if not np.isnan(test_sharpe_by_lb[k]) else -np.inf))

    test_equity = (1 + series_by_lb[lb_star].iloc[split:]).cumprod()
    test_dd = (test_equity / test_equity.cummax() - 1).min()
    test_total_return = test_equity.iloc[-1] - 1

    # Out-of-sample p_pass: bootstrap source restricted to the test bar range only, lb
    # fixed at the train-selected value, leverage=1.0 (plain vol-targeted strategy, no
    # overlay -- see tsmom_barrier_sim's session-14 correction). bar_range indexes into
    # extract_market_tuples' post-dropna space, which starts one bar later than df's raw
    # index (the first row has no return) -- series_by_lb is aligned the same way
    # (df.index[1:]) via _strat_ret_series, so `split`/`len` computed there line up
    # directly with the bar_range passed to the barrier sim.
    bar_range = (split, n_bars_series)
    sim_kwargs = {} if round_trip_cost is None else {"round_trip_cost": round_trip_cost}
    sim = run_barrier_sim(coin, interval, lb_star, n_paths=400, cppi=False,
                           leverage=1.0, bar_range=bar_range, **sim_kwargs)

    return {
        "coin": coin, "interval": interval, "lb_star": lb_star,
        "train_sharpe_at_lb_star": train_sharpe[lb_star],
        "test_sharpe_at_lb_star": test_sharpe_at_lbstar,
        "lb_test_optimal": lb_test_optimal,
        "test_sharpe_at_test_optimal": test_sharpe_by_lb[lb_test_optimal],
        "test_max_dd": test_dd, "test_total_return": test_total_return,
        "test_n_bars": n_bars_series - split, "bar_range": bar_range,
        "oos_leverage_k": sim["leverage"], "oos_p_pass": sim["pass"],
        "oos_fail_static": sim["fail_static"], "oos_fail_daily": sim["fail_daily"],
        "oos_unresolved": sim["unresolved"],
    }


def _strat_ret_series(coin: str, interval: str, lookback_bars: int,
                       round_trip_cost: float | None = None) -> pd.Series:
    """Re-derives backtest_trailing_stop's per-bar strategy-return series (that
    function currently returns only summary stats) by calling its exact same logic
    via a local re-import -- kept as a thin duplication-avoider rather than editing
    tsmom_hypothesis.py's return contract, since other code already depends on it."""
    import tsmom_hypothesis as m
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

    close, high, low = df["c"].to_numpy(), df["h"].to_numpy(), df["l"].to_numpy()
    mom_a, atr_a, vol_a = mom.to_numpy(), atr.to_numpy(), realized_vol.to_numpy()
    n = len(df)
    strat_ret = np.zeros(n)
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
    return pd.Series(strat_ret, index=df.index).iloc[1:]


if __name__ == "__main__":
    rows = [walk_forward_one(coin, interval) for coin, interval in COMBOS]
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_json(OUT / "tsmom_walkforward.json", orient="records", indent=2)

    fmt = {
        "train_sharpe_at_lb_star": "{:.2f}".format, "test_sharpe_at_lb_star": "{:.2f}".format,
        "test_sharpe_at_test_optimal": "{:.2f}".format, "test_max_dd": "{:.1%}".format,
        "test_total_return": "{:.1%}".format, "oos_leverage_k": "{:.3f}".format,
        "oos_p_pass": "{:.1%}".format, "oos_fail_static": "{:.1%}".format,
        "oos_fail_daily": "{:.1%}".format, "oos_unresolved": "{:.1%}".format,
    }
    print(df.to_string(index=False, formatters=fmt))
    print(f"\nSession 10 in-sample headline p_pass (for comparison, NOT walk-forward): "
          f"BTC 4h=24.5%, ETH 12h=27.5%, SOL 12h=28.5%")
    print(f"\nSelection-artifact gap: mean(train Sharpe) - mean(test Sharpe @ lb*) = "
          f"{df['train_sharpe_at_lb_star'].mean() - df['test_sharpe_at_lb_star'].mean():.2f}")
