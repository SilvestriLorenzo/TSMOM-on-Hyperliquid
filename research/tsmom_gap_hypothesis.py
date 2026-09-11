"""
H_TSMOM_GAP (session 20, user-proposed: "something faster than the current strategy").

The production H_TSMOM signal is locked at 4h/8h/12h (session 13/14). Session 18 tested
the opposite extreme -- 1m/5m -- and found it catastrophically negative (Sharpe -9 to
-540), because BTC/ETH/SOL returns are reversal-dominated at that frequency and turnover
approaches 100%/bar. Neither prior test touched the band in between: 15m/30m/1h/2h.
That's a real, previously-untested gap, not a re-run of either prior result -- this
answers "does the reversal-dominance found at 1-5min already extend into the 15min-2h
band, or does TSMOM's edge appear somewhere before 4h" empirically, rather than assuming
either extrapolation.

Reuses `tsmom_hypothesis.py`'s exact validated backtest/backtest_trailing_stop functions
unmodified (same vol-targeting, same ATR-14/2.5x trailing stop, same leverage caps, same
9bps cost convention) -- only the interval grid and BARS_PER_YEAR annualization table are
extended, so this is a like-for-like comparison against the already-locked 4h/8h/12h
numbers, not a differently-built strategy.
"""
import numpy as np
import pandas as pd

from research import tsmom_hypothesis as th
from research.tsmom_walkforward import _strat_ret_series, sharpe
# tsmom_walkforward.py inserts research/ onto sys.path and _strat_ret_series does its own
# bare `import tsmom_hypothesis as m` -- that resolves to a SEPARATE module object from
# `research.tsmom_hypothesis` (different sys.modules key, same file), so both copies'
# BARS_PER_YEAR need the extended interval band or _strat_ret_series KeyErrors.
import tsmom_hypothesis as th_bare

GAP_INTERVALS = ["15m", "30m", "1h", "2h"]
_EXTRA_BARS_PER_YEAR = {
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "2h": 365 * 12,
}
th.BARS_PER_YEAR.update(_EXTRA_BARS_PER_YEAR)
th_bare.BARS_PER_YEAR.update(_EXTRA_BARS_PER_YEAR)


EXCLUDE_EPISODE = (pd.Timestamp("2026-08-17"), pd.Timestamp("2026-08-27"))


def outlier_excision_check(coin: str, interval: str, lookback_bars: int) -> dict:
    """Direct test of whether H_TSMOM_GAP's positive 30m-2h cells are real or entirely
    the Aug 17-27 rally episode (same outlier-diagnosis discipline as H_ETF and
    H_LEADLAG): compute the full-sample ATR-stop return series, then compare Sharpe
    on (a) the whole series, (b) the series with the flagged episode's bars removed,
    and (c) the flagged episode alone."""
    bars_per_year = th.BARS_PER_YEAR[interval]
    full = _strat_ret_series(coin, interval, lookback_bars)
    in_episode = (full.index >= EXCLUDE_EPISODE[0]) & (full.index < EXCLUDE_EPISODE[1])
    ex_episode = full[~in_episode]
    episode_only = full[in_episode]
    return {
        "coin": coin, "interval": interval, "lb": lookback_bars,
        "sharpe_full": sharpe(full, bars_per_year),
        "sharpe_ex_episode": sharpe(ex_episode, bars_per_year),
        "sharpe_episode_only": sharpe(episode_only, bars_per_year) if len(episode_only) > 2 else np.nan,
        "n_episode_bars": int(in_episode.sum()), "n_total_bars": len(full),
        "total_return_full": (1 + full).prod() - 1,
        "total_return_ex_episode": (1 + ex_episode).prod() - 1,
        "total_return_episode_only": (1 + episode_only).prod() - 1 if len(episode_only) else np.nan,
    }


def walk_forward_sharpe_only(coin: str, interval: str, train_frac: float = 0.7) -> dict:
    """Lighter than tsmom_walkforward.walk_forward_one -- Sharpe-only train/test split,
    no barrier-sim p_pass (that machinery's BARS_PER_DAY/BLOCK_LEN/HORIZON_BARS_1095D
    tables are hardcoded to 4h/8h/12h in tsmom_barrier_sim.py; not worth extending them
    to a new interval band before the Sharpe-only gate says it's worth the effort)."""
    bars_per_year = th.BARS_PER_YEAR[interval]
    series_by_lb = {lb: _strat_ret_series(coin, interval, lb) for lb in [1, 2, 3]}
    n = len(next(iter(series_by_lb.values())))
    split = int(n * train_frac)
    train_sharpe = {lb: sharpe(s.iloc[:split], bars_per_year) for lb, s in series_by_lb.items()}
    lb_star = max(train_sharpe, key=lambda k: (train_sharpe[k] if not np.isnan(train_sharpe[k]) else -np.inf))
    test_series = series_by_lb[lb_star].iloc[split:]
    test_sharpe = sharpe(test_series, bars_per_year)
    equity = (1 + test_series).cumprod()
    return {
        "coin": coin, "interval": interval, "lb_star": lb_star,
        "train_sharpe": train_sharpe[lb_star], "test_sharpe": test_sharpe,
        "test_n_bars": len(test_series), "test_total_return": equity.iloc[-1] - 1,
        "test_max_dd": (equity / equity.cummax() - 1).min(),
    }

fmt = {
    "years": "{:.2f}".format, "ann_return": "{:.1%}".format, "ann_vol": "{:.1%}".format,
    "sharpe": "{:.2f}".format, "max_dd": "{:.1%}".format,
    "avg_leverage": "{:.2f}".format, "total_return": "{:.1%}".format,
}

if __name__ == "__main__":
    coins = ["BTC", "ETH", "SOL"]

    print(f"=== H_TSMOM_GAP naive: re-flip every bar, no trailing stop, {GAP_INTERVALS} ===")
    rows = [th.backtest(coin, interval, lb) for coin in coins for interval in GAP_INTERVALS
            for lb in [1, 2, 3]]
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, formatters=fmt))
    print("\nBest by Sharpe:")
    print(res.sort_values("sharpe", ascending=False).head(8).to_string(index=False, formatters=fmt))

    print(f"\n=== H_TSMOM_GAP ATR trailing-stop exit, {GAP_INTERVALS} ===")
    rows2 = [th.backtest_trailing_stop(coin, interval, lb) for coin in coins
             for interval in GAP_INTERVALS for lb in [1, 2, 3]]
    res2 = pd.DataFrame(rows2)
    print(res2.to_string(index=False, formatters=fmt))
    print("\nBest by Sharpe:")
    print(res2.sort_values("sharpe", ascending=False).head(8).to_string(index=False, formatters=fmt))

    print(f"\nReference points: locked 4h/8h/12h production combos (session 13/14 walk-forward "
          f"OOS p_pass, not this script's in-sample Sharpe) -- BTC 4h, ETH 8h, ETH 12h, SOL 12h, "
          f"all ATR-trailing-stop, Sharpe range 0.57-1.95 OOS. H_TSMOM_FAST (session 18): 1m/5m, "
          f"same mechanism, Sharpe -9 to -540.")

    print(f"\n=== H_TSMOM_GAP walk-forward (Sharpe-only gate, 70/30 chronological split, "
          f"lb selected on train), {GAP_INTERVALS} ===")
    wf_rows = [walk_forward_sharpe_only(coin, interval) for coin in coins for interval in GAP_INTERVALS]
    wf = pd.DataFrame(wf_rows)
    wf_fmt = {"train_sharpe": "{:.2f}".format, "test_sharpe": "{:.2f}".format,
              "test_total_return": "{:.1%}".format, "test_max_dd": "{:.1%}".format}
    print(wf.to_string(index=False, formatters=wf_fmt))

    print(f"\n=== H_TSMOM_GAP outlier excision (Aug 17-27 rally episode), "
          f"30m/1h/2h, all 3 lookbacks, all 3 coins ===")
    exc_rows = [outlier_excision_check(coin, interval, lb) for coin in coins
                for interval in ["30m", "1h", "2h"] for lb in [1, 2, 3]]
    exc = pd.DataFrame(exc_rows)
    exc_fmt = {"sharpe_full": "{:.2f}".format, "sharpe_ex_episode": "{:.2f}".format,
               "sharpe_episode_only": "{:.2f}".format, "total_return_full": "{:.1%}".format,
               "total_return_ex_episode": "{:.1%}".format, "total_return_episode_only": "{:.1%}".format}
    print(exc.to_string(index=False, formatters=exc_fmt))
