"""
H_TSMOM_FAST: same exact methodology as H_TSMOM (research/tsmom_hypothesis.py) --
sign of the most recent bar's return, vol-targeted position sizing to a 20% annualized
target, leverage capped per PROPR.md, optional ATR trailing-stop exit -- just run at
1m/5m bars instead of 4h/8h/12h. Answers a different question from H_IMPULSE: that
hypothesis tested a *new* mechanic (confirmed impulse -> short fixed-hold continuation,
ported from a Polymarket skill repo) and found reversal, not continuation, at this
frequency. This one asks whether the strategy we already run and trust at 4h/8h/12h
still has an edge if you just point its own unmodified signal/sizing/exit logic at a
faster bar.

Deliberately imports tsmom_hypothesis's tested functions unmodified (load_candles,
backtest, backtest_trailing_stop, ROUND_TRIP_COST, LEVERAGE_CAP, TARGET_VOL_ANNUAL,
VOL_LOOKBACK_BARS) rather than duplicating the logic, and only *adds* "1m"/"5m" keys to
its BARS_PER_YEAR dict at runtime in this process -- tsmom_beta_live.py (the live cron
bot) imports the same module but runs as its own separate process from the on-disk file,
which this does not touch, so this is zero-risk to the running bot.

Same data caveat as H_IMPULSE: Hyperliquid's candleSnapshot only retains ~3.6 days of 1m
and ~17.5 days of 5m history -- a single thin recent window, not independent multi-year
draws. Preliminary directional read only.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m  # noqa: E402

m.BARS_PER_YEAR["1m"] = 365 * 24 * 60
m.BARS_PER_YEAR["5m"] = 365 * 24 * 12

GRID = {"1m": [1, 3, 5, 10, 20, 30], "5m": [1, 2, 3, 6, 12]}


if __name__ == "__main__":
    coins = sys.argv[1:] or ["BTC", "ETH", "SOL"]
    fmt = {
        "years": "{:.3f}".format, "ann_return": "{:.1%}".format, "ann_vol": "{:.1%}".format,
        "sharpe": "{:.2f}".format, "max_dd": "{:.1%}".format,
        "avg_leverage": "{:.2f}".format, "total_return": "{:.1%}".format,
    }

    for interval in ["1m", "5m"]:
        print(f"\n=== H_TSMOM_FAST naive: re-flip every bar, interval={interval} ===")
        rows = [m.backtest(coin, interval, lb) for coin in coins for lb in GRID[interval]]
        res = pd.DataFrame(rows)
        print(res.to_string(index=False, formatters=fmt))
        print(f"\nBest by Sharpe:")
        print(res.sort_values("sharpe", ascending=False).head(5).to_string(index=False, formatters=fmt))

        print(f"\n=== H_TSMOM_FAST ATR trailing-stop exit, interval={interval} ===")
        rows2 = [m.backtest_trailing_stop(coin, interval, lb) for coin in coins for lb in GRID[interval]]
        res2 = pd.DataFrame(rows2)
        print(res2.to_string(index=False, formatters=fmt))
        print(f"\nBest by Sharpe:")
        print(res2.sort_values("sharpe", ascending=False).head(5).to_string(index=False, formatters=fmt))

    print(f"\nReference: locked H_TSMOM production config (4h/8h/12h, session 13/14) "
          f"and Bui & Nguyen's Binance benchmark (Sharpe 1.83, MDD -16.1%) are the honest "
          f"comparison points -- not each other's in-sample-cherry-picked best cell.")
