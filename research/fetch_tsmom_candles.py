"""
Fetches native sub-daily candles for the vol-targeted TSMOM hypothesis (user-proposed,
session 10 continuation): a 4-12h signal horizon on BTC/ETH/SOL. Hyperliquid's
candleSnapshot has no native "6h" interval (422 on that string) -- the available
intervals bracketing the requested 4-12h band are 4h, 8h, and 12h, so those three are
fetched instead of resampling from a shorter-history base interval. Reuses
fetch_reference_data.fetch_candles(); each interval gets its own 5000-candle-capped
request, which buys very different amounts of history per interval (4h -> ~2.3y,
8h -> ~4.6y, 12h -> ~6.8y) -- coarser intervals aren't just "smoother", they're also
the only ones with enough real sample length to trust a backtest on.
"""
import json
import time
from pathlib import Path

from research.fetch_reference_data import fetch_candles

OUT = Path(__file__).parent / "output"

if __name__ == "__main__":
    for coin in ["BTC", "ETH", "SOL"]:
        for interval in ["4h", "8h", "12h"]:
            candles = fetch_candles(coin, interval=interval, lookback_days=3000)
            path = OUT / f"{coin}_{interval}_candles.json"
            path.write_text(json.dumps(candles))
            t0, t1 = candles[0]["t"], candles[-1]["t"]
            days = (t1 - t0) / 86_400_000
            print(f"{coin} {interval}: {len(candles)} candles, {days:.0f} days "
                  f"({time.strftime('%Y-%m-%d', time.gmtime(t0/1000))} -> "
                  f"{time.strftime('%Y-%m-%d', time.gmtime(t1/1000))})")
            time.sleep(1)
