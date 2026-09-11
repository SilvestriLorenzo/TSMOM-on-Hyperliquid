"""
H_TSMOM: user-proposed (session 10) vol-targeted time-series momentum on BTC/ETH/SOL at
a 4-12h signal horizon. Tests the RAW signal first, before any Magdon-Ismail-Atiya
expected-max-drawdown sizing or Grossman-Zhou dynamic de-risking overlay -- same
discipline as every other hypothesis here (H_ETF, NATGAS, OI-divergence): confirm the
underlying signal survives real costs before building sophisticated risk machinery on
top of a signal that might not have edge at all.

Reference point: Bui & Nguyen (arXiv:2602.11708) benchmark their "Vol-Scaled TSMOM"
baseline (10% annualized vol target, no market-cap filter, no asymmetric allocation, no
ATR trailing stop) at Sharpe 1.83, MDD -16.1% on Binance Futures, 2022-2024. That's the
honest comparison point for this simpler single/few-asset version -- not their
full-framework 2.41, which depends on components (150-coin cross-sectional selection,
70/30 long-short, monthly reoptimization) this hypothesis doesn't include.

Signal: sign of the single most recent bar's return (classic TSMOM at native bar
resolution -- Hyperliquid has no 6h interval, so 4h/8h/12h bracket the requested 4-12h
band instead of resampling from a shorter-history base). Position scaled to a target
per-bar volatility using trailing realized vol, capped at Propr's per-asset-class
leverage limit (BTC/ETH 5x, SOL "other crypto" 2x per PROPR.md).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"
ROUND_TRIP_COST = 0.0009  # 9bps, same core-dex taker convention used throughout this project
LEVERAGE_CAP = {"BTC": 5.0, "ETH": 5.0, "SOL": 2.0}
TARGET_VOL_ANNUAL = 0.20
BARS_PER_YEAR = {"4h": 365 * 6, "8h": 365 * 3, "12h": 365 * 2}
VOL_LOOKBACK_BARS = 20


def load_candles(coin: str, interval: str) -> pd.DataFrame:
    data = json.loads((OUT / f"{coin}_{interval}_candles.json").read_text())
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    for col in ("o", "h", "l", "c"):
        df[col] = df[col].astype(float)
    return df[["t", "o", "h", "l", "c"]].set_index("t")


def backtest_trailing_stop(coin: str, interval: str, lookback_bars: int = 1,
                            atr_k: int = 14, atr_mult: float = 2.5) -> dict:
    """Same entry signal as backtest() (sign of momentum over lookback_bars, no
    threshold), replacing the every-bar re-flip with Bui & Nguyen's ATR trailing-stop
    exit (their single biggest ablation contributor): once in a position, hold until a
    stop that only ratchets in the favorable direction is breached, then wait for the
    next momentum-sign entry. Position size fixed at entry (vol-target scaled,
    leverage-capped), not re-scaled mid-trade.

    Timing, deliberately conservative to avoid look-ahead: the stop level checked
    *during* bar t is fixed using only data through bar t-1's close (both the ATR and
    the ratchet). A breach is detected against bar t's own intrabar low/high (real data
    we have from OHLC, not the close), and the bar's return is truncated to the stop
    price rather than the full bar -- crediting the loss actually taken up to the stop,
    not silently dodging the rest of an adverse bar the way checking the stop against
    that same bar's own close (only knowable at the close) would."""
    df = load_candles(coin, interval)
    log_ret = np.log1p(df["c"].pct_change())
    mom = df["c"].pct_change(lookback_bars).shift(1)
    tr = np.maximum(df["h"] - df["l"], np.maximum((df["h"] - df["c"].shift(1)).abs(),
                                                    (df["l"] - df["c"].shift(1)).abs()))
    atr = tr.rolling(atr_k).mean().shift(1)  # known as of t-1, used to size bar t's stop
    realized_vol = log_ret.rolling(VOL_LOOKBACK_BARS).std().shift(1)
    bars_per_year = BARS_PER_YEAR[interval]
    target_vol_per_bar = TARGET_VOL_ANNUAL / np.sqrt(bars_per_year)

    close = df["c"].to_numpy()
    high = df["h"].to_numpy()
    low = df["l"].to_numpy()
    mom_a = mom.to_numpy()
    atr_a = atr.to_numpy()
    vol_a = realized_vol.to_numpy()
    n = len(df)

    strat_ret = np.zeros(n)
    pos_size, pos_dir, stop_level = 0.0, 0, np.nan
    n_trades = 0

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            # flat entering bar t: decide using pre-t info only, apply full bar return
            # (same convention as the naive backtest -- no look-ahead)
            if (not np.isnan(mom_a[t]) and mom_a[t] != 0 and not np.isnan(atr_a[t])
                    and not np.isnan(vol_a[t]) and vol_a[t] > 0):
                pos_dir = 1 if mom_a[t] > 0 else -1
                pos_size = min(target_vol_per_bar / vol_a[t], LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr_a[t]
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * ROUND_TRIP_COST / 2
                n_trades += 1
        else:
            # holding into bar t: stop_level was fixed using data through t-1 only.
            # check breach against this bar's own intrabar extreme, truncate the
            # captured return to the stop price if breached.
            breached = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            if breached:
                exit_px = stop_level
                strat_ret[t] = pos_dir * pos_size * np.log(exit_px / prev_close) - pos_size * ROUND_TRIP_COST / 2
                pos_dir, pos_size, stop_level = 0, 0.0, np.nan
                n_trades += 1
            else:
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr_a[t]):
                    if pos_dir == 1:
                        stop_level = max(stop_level, close[t] - atr_mult * atr_a[t])
                    else:
                        stop_level = min(stop_level, close[t] + atr_mult * atr_a[t])

    strat_ret = pd.Series(strat_ret, index=df.index).iloc[1:]
    equity = (1 + strat_ret).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    ann_return = strat_ret.mean() * bars_per_year
    ann_vol = strat_ret.std() * np.sqrt(bars_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    return {
        "coin": coin, "interval": interval, "lookback_bars": lookback_bars,
        "n_bars": len(strat_ret), "years": len(strat_ret) / bars_per_year,
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_dd": drawdown.min(), "total_return": equity.iloc[-1] - 1, "n_trades": n_trades,
    }


def backtest(coin: str, interval: str, lookback_bars: int = 1) -> dict:
    df = load_candles(coin, interval)
    ret = df["c"].pct_change()
    log_ret = np.log1p(ret)

    mom = df["c"].pct_change(lookback_bars)
    signal = np.sign(mom).shift(1)  # decided at close of bar t, applied to bar t+1's return
    realized_vol = log_ret.rolling(VOL_LOOKBACK_BARS).std().shift(1)
    bars_per_year = BARS_PER_YEAR[interval]
    target_vol_per_bar = TARGET_VOL_ANNUAL / np.sqrt(bars_per_year)
    scale = (target_vol_per_bar / realized_vol).clip(upper=LEVERAGE_CAP[coin])
    position = (signal * scale).fillna(0.0)

    turnover = position.diff().abs().fillna(0.0)
    cost = turnover * ROUND_TRIP_COST / 2
    strat_ret = (position * log_ret) - cost
    strat_ret = strat_ret.dropna()

    equity = (1 + strat_ret).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    ann_return = strat_ret.mean() * bars_per_year
    ann_vol = strat_ret.std() * np.sqrt(bars_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    return {
        "coin": coin, "interval": interval, "lookback_bars": lookback_bars,
        "n_bars": len(strat_ret), "years": len(strat_ret) / bars_per_year,
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_dd": drawdown.min(), "avg_leverage": position.abs().mean(),
        "total_return": equity.iloc[-1] - 1,
    }


if __name__ == "__main__":
    coins = sys.argv[1:] or ["BTC", "ETH", "SOL"]
    fmt = {
        "years": "{:.2f}".format, "ann_return": "{:.1%}".format, "ann_vol": "{:.1%}".format,
        "sharpe": "{:.2f}".format, "max_dd": "{:.1%}".format,
        "avg_leverage": "{:.2f}".format, "total_return": "{:.1%}".format,
    }

    print("=== Naive: re-flip every bar, no trailing stop ===")
    rows = [backtest(coin, interval, lb) for coin in coins for interval in ["4h", "8h", "12h"]
            for lb in [1, 2, 3]]
    res = pd.DataFrame(rows)
    print(res.to_string(index=False, formatters=fmt))
    print(f"\nBest by Sharpe:")
    print(res.sort_values("sharpe", ascending=False).head(5).to_string(index=False, formatters=fmt))

    print("\n=== ATR trailing-stop exit (Bui & Nguyen's mechanism, alpha=2.5, ATR-14) ===")
    rows2 = [backtest_trailing_stop(coin, interval, lb) for coin in coins
              for interval in ["4h", "8h", "12h"] for lb in [1, 2, 3]]
    res2 = pd.DataFrame(rows2)
    print(res2.to_string(index=False, formatters=fmt))
    print(f"\nBest by Sharpe:")
    print(res2.sort_values("sharpe", ascending=False).head(5).to_string(index=False, formatters=fmt))

    print(f"\nReference: Bui & Nguyen's Vol-Scaled TSMOM baseline (Binance, 2022-2024, "
          f"single crypto benchmark shape, not this project's exact assets/venue/costs): "
          f"Sharpe 1.83, MDD -16.1%.")
