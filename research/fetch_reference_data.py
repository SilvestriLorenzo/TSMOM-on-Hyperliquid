"""
Phase 1 data pull. Independent of ~/hyperliquid-perps's collected data — hits
Hyperliquid's free public info API directly for BTC/ETH daily candles (full available
history) + full funding history, to ground alpha hypotheses in real numbers. No auth
needed (info endpoints), respects the documented IP-level budget (~1,200 weight
units/min) by doing a handful of one-off calls, not polling.
"""
import time
import httpx

API = "https://api.hyperliquid.xyz/info"
DAY_MS = 86_400_000


def post(body: dict) -> object:
    r = httpx.post(API, json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_candles(coin: str, interval: str = "1d", lookback_days: int = 3000) -> list[dict]:
    """Default interval is daily, deliberately. Originally documented as "candleSnapshot
    is capped at 5000 candles per request" -- corrected session 20: it's actually a
    server-side RETENTION limit at sub-daily intervals (verified directly: a 1h request
    for a 30-day window ~500 days ago returns 0 candles, while the identical window at
    1d resolution returns a full 31), not a per-call size cap. `fetch_candles_paginated`
    confirmed this the hard way -- chunked requests return the exact same ~5000-candle
    window as a single call, because there's nothing older to page into. A coarser
    interval buys more real history for a structural reason (daily candles are kept
    indefinitely; 30m/1h/2h/etc. are only kept for their own most recent ~5000 bars), not
    just a fixed-candle-count-per-call artifact -- there's no way to get more sub-daily
    history from this endpoint, paginated or not."""
    now = int(time.time() * 1000)
    start = now - lookback_days * DAY_MS
    return post({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval, "startTime": start, "endTime": now},
    })


INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": DAY_MS,
}


def fetch_candles_paginated(coin: str, interval: str, lookback_days: int) -> list[dict]:
    """Session 20: built to work around what was believed to be a per-call 5000-candle
    cap, hoping to get a longer, cleaner 30m/1h/2h history than the single-call ~104/
    208/417-day windows (thin enough that a single 1-2 week episode dominated a
    walk-forward test split -- H_TSMOM_GAP's ETH 1h/2h test windows were both entirely
    explained by one rally episode already flagged fragile in H_ETF). **Doesn't actually
    help**: verified directly that requesting an old window at 1h resolution returns 0
    candles while the identical window at 1d resolution returns real data -- this is a
    server-side retention limit at sub-daily intervals, not a per-request cap, so no
    amount of pagination reaches data that was never retained. Kept in the codebase as
    the documented negative result (and it's still correct/useful if Hyperliquid ever
    extends sub-daily retention), but don't expect it to return more than a single
    `fetch_candles` call already does today."""
    bar_ms = INTERVAL_MS[interval]
    chunk_ms = 4900 * bar_ms
    now = int(time.time() * 1000)
    cursor = now - lookback_days * DAY_MS
    out: list[dict] = []
    seen_t: set[int] = set()
    while cursor < now:
        end = min(cursor + chunk_ms, now)
        page = post({
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval, "startTime": cursor, "endTime": end},
        })
        for c in page:
            if c["t"] not in seen_t:
                seen_t.add(c["t"])
                out.append(c)
        cursor = end + bar_ms  # avoid re-requesting the boundary candle
        time.sleep(0.3)
    out.sort(key=lambda c: c["t"])
    return out


def fetch_funding(coin: str, lookback_days: int = 3000) -> list[dict]:
    now = int(time.time() * 1000)
    start = now - lookback_days * DAY_MS
    out: list[dict] = []
    cursor = start
    while True:
        page = post({"type": "fundingHistory", "coin": coin, "startTime": cursor, "endTime": now})
        if not page:
            break
        out.extend(page)
        last_t = page[-1]["time"]
        if last_t <= cursor or len(page) < 500:
            break
        cursor = last_t + 1
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    for coin in ["BTC", "ETH"]:
        candles = fetch_candles(coin)
        (out_dir / f"{coin}_1d_candles.json").write_text(json.dumps(candles))
        print(f"{coin}: {len(candles)} 1d candles, "
              f"{candles[0]['t'] if candles else None} -> {candles[-1]['t'] if candles else None}")

        funding = fetch_funding(coin)
        (out_dir / f"{coin}_funding.json").write_text(json.dumps(funding))
        print(f"{coin}: {len(funding)} funding records")
