"""
H_POST_TP_SCORE_V2 (session 26): two new second-stage score types layered on the
corrected H_POST_TP_REENTRY mechanism, independently researched (web search, not just
recalled from memory) after the corrected session-25 mechanism came back roughly NEUTRAL
rather than a clear reject -- a legitimate foundation to refine. Motivation: the
corrected mechanism's own walk-forward search converges on very short (2-5 bar) price
lookbacks for 3 of 4 combos, landing in the same horizon H_TSMOM_FAST (session 18)
already showed has no pure-price-momentum edge. Re-scoring the same price series at an
even shorter horizon is unlikely to do better -- a genuinely DIFFERENT information
channel might.

(a) Volume/participation confirmation, Chaikin-Money-Flow-style: every candle file this
project has ever loaded carries `v` (volume) and `n` (trade count) fields that
`tsmom_hypothesis.load_candles` silently drops and no script here has ever used. Money
flow multiplier = ((close-low)-(high-close))/(high-low) (where the bar's close sat
within its own range), weighted by volume and averaged over a short window -- a reading
of whether the move to TP was accompanied by genuine participation (continuation-like)
or thinning volume (exhaustion-like), independent of the price-return score.

(b) A Hurst-exponent-style trend-persistence regime FILTER (not a directional score by
itself): gates the hold/flip decision on whether the recent price path is actually in a
persistent/trending regime (generalized-Hurst estimate > 0.5) vs. mean-reverting/choppy
(<= 0.5) -- forces flat instead of acting on a hold/flip signal when the regime doesn't
support it. Implemented as a lightweight variance-ratio estimator (Peters/di Matteo
style: Var(q-bar return) scales as q^(2H) under a self-affine model, so H = 0.5 +
0.5*log(VR)/log(q) where VR = Var(q-bar return)/(q*Var(1-bar return))), fast enough to
precompute as a rolling array rather than needing a full R/S regression per bar.

CMF needs volume resampled ALONGSIDE price in the Monte Carlo bootstrap (a synthetic
path's volume must correspond to its own price action, not be independent noise) --
extends tsmom_barrier_sim's market-tuple/bootstrap machinery with a volume channel
(extract_market_tuples_v2/block_bootstrap_path_v2), duplicated rather than modifying the
validated originals, matching this project's established convention. The Hurst filter is
purely price-based and needs no such extension.

Data-availability note (from the independent research): order-flow-imbalance,
funding/OI crowding, and liquidation-cascade signals are, in the literature, more
directly analogous to "short-term momentum" in the microstructure sense -- but
Hyperliquid's own live collector (`src/collector/`) has only ~9 days of L2/trade/
asset_ctx history as of this session (restarted session 20/21), far short of what a
walk-forward test to this project's own rigor bar would need. Flagged for later, not
attempted here; CMF and the Hurst filter are both immediately testable on the full
multi-year candle history already on disk.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tsmom_hypothesis as m
from tsmom_barrier_sim import BLOCK_LEN, TARGET, STATIC_FLOOR, DAILY_LOSS
from tsmom_joint_portfolio_barrier_sim import LOCKED, WEIGHT, BLOCK_LEN_DAYS, N_SIMS, CYCLE_CAP_DAYS, simulate_pooled_path
from tsmom_exit_variants import _atr_series, ATR_K, TRAIN_FRAC, vol_target_returns as _baseline_vtr
from tsmom_post_tp_reentry import TP_ATR_MULT_BY_COMBO, ATR_MULT_BASE, SCORE_LOOKBACK_GRID, EPS_GRID
from tsmom_barrier_sim import extract_market_tuples

OUT = Path(__file__).parent / "output"
BARS_PER_DAY = {"4h": 6, "8h": 3, "12h": 2}
N_PATHS = 400
HORIZON_DAYS = 1095

HURST_WINDOW = 30
HURST_Q = 5
HURST_THRESHOLD = 0.5

CMF_LOOKBACK_GRID = [5, 10, 20]
CMF_EPS_GRID = [0.0, 0.1, 0.2]


def load_candles_v2(coin: str, interval: str) -> pd.DataFrame:
    """Duplicates tsmom_hypothesis.load_candles but keeps `v` (volume) and `n` (trade
    count), which the validated loader silently drops."""
    data = json.loads((m.OUT / f"{coin}_{interval}_candles.json").read_text())
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    for col in ("o", "h", "l", "c", "v"):
        df[col] = df[col].astype(float)
    df["n"] = df["n"].astype(int)
    return df[["t", "o", "h", "l", "c", "v", "n"]].set_index("t")


def extract_market_tuples_v2(coin: str, interval: str, bar_range: tuple[int, int] | None = None):
    """Duplicates tsmom_barrier_sim.extract_market_tuples, adding a volume channel
    aligned the same way (same dropna, same optional bar_range slice)."""
    df = load_candles_v2(coin, interval)
    log_ret = np.log(df["c"] / df["c"].shift(1))
    high_frac = np.log(df["h"] / df["c"])
    low_frac = np.log(df["l"] / df["c"])
    valid = log_ret.notna()
    log_ret_a = log_ret[valid].to_numpy()
    high_a = high_frac[valid].to_numpy()
    low_a = low_frac[valid].to_numpy()
    vol_a = df["v"][valid].to_numpy()
    if bar_range is not None:
        start, end = bar_range
        log_ret_a, high_a, low_a, vol_a = log_ret_a[start:end], high_a[start:end], low_a[start:end], vol_a[start:end]
    return log_ret_a, high_a, low_a, vol_a


def block_bootstrap_path_v2(log_ret, high_frac, low_frac, volume, n_bars, block_len, rng):
    """Duplicates tsmom_barrier_sim.block_bootstrap_path, resampling volume in the SAME
    blocks as price so a synthetic path's volume stays attached to its own price
    action."""
    n_hist = len(log_ret)
    out_ret, out_hi, out_lo, out_vol = [], [], [], []
    covered = 0
    while covered < n_bars:
        start = rng.integers(0, n_hist)
        idx = (start + np.arange(block_len)) % n_hist
        out_ret.append(log_ret[idx]); out_hi.append(high_frac[idx])
        out_lo.append(low_frac[idx]); out_vol.append(volume[idx])
        covered += block_len
    return (np.concatenate(out_ret)[:n_bars], np.concatenate(out_hi)[:n_bars],
            np.concatenate(out_lo)[:n_bars], np.concatenate(out_vol)[:n_bars])


def _cmf_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """Chaikin-Money-Flow-style score, evaluated AT each bar t (not shifted -- by the
    time a TP-fire decision is made, bar t is fully realized, matching the "fresh score
    at TP-fire bar" convention this project already established). Bounded in [-1, 1]:
    +1 = every bar in the window closed at its high with volume behind it (strong
    accumulation), -1 = the mirror image (strong distribution)."""
    rng_ = high - low
    mfm = np.where(rng_ > 0, ((close - low) - (high - close)) / np.where(rng_ > 0, rng_, 1.0), 0.0)
    mfv = pd.Series(mfm * volume)
    vol_s = pd.Series(volume)
    return (mfv.rolling(window).sum() / vol_s.rolling(window).sum()).to_numpy()


def _hurst_like_array(close: np.ndarray, window: int, q: int) -> np.ndarray:
    """Variance-ratio Hurst estimate: under a self-affine (fractional-Brownian-motion-
    like) model Var(q-bar cumulative return) scales as q^(2H), so
    H = 0.5 + 0.5*log(VR)/log(q) where VR = Var(q-bar return)/(q*Var(1-bar return)).
    H>0.5 = trending/persistent, H<0.5 = mean-reverting/anti-persistent, H=0.5 = random
    walk. A lightweight rolling approximation, not a full R/S regression -- adequate for
    a regime GATE, not intended as a precise Hurst estimator."""
    close_s = pd.Series(close)
    r1 = np.log(close_s / close_s.shift(1))
    rq = np.log(close_s / close_s.shift(q))
    var1 = r1.rolling(window).var()
    varq = rq.rolling(window).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = varq / (q * var1)
        h = 0.5 + 0.5 * np.log(vr) / np.log(q)
    return h.to_numpy()


def vol_target_returns_post_tp_v2(log_ret, high, low, close, volume, lookback_bars, atr_k, coin, interval,
                                   tp_atr_mult, score_type, score_lookback, eps, hurst_gate,
                                   atr_mult=ATR_MULT_BASE, hurst_window=HURST_WINDOW, hurst_q=HURST_Q,
                                   hurst_threshold=HURST_THRESHOLD, round_trip_cost=m.ROUND_TRIP_COST):
    """Same state machine as tsmom_post_tp_reentry.vol_target_returns_post_tp (base
    trade watches ATR stop + TP; TP-fire triggers a second decision; re-entry/flip
    trades watch only the ATR stop, no chaining), extended with score_type in
    {"raw","vol_norm","cmf"} and an independent hurst_gate that can force any hold/flip
    signal to flat when the regime filter doesn't confirm a trending regime. Correctly
    makes every score relative to pos_dir (pos_dir * chosen) before thresholding -- the
    sign-convention bug found and fixed in session 26's correction to
    tsmom_post_tp_reentry.py."""
    n = len(close)
    mom = pd.Series(close).pct_change(lookback_bars).shift(1).to_numpy()
    atr = _atr_series(high, low, close, atr_k)
    log_ret_padded = np.concatenate([[np.nan], log_ret])
    realized_vol = pd.Series(log_ret_padded).rolling(m.VOL_LOOKBACK_BARS).std().shift(1).to_numpy()
    target_vol_per_bar = m.TARGET_VOL_ANNUAL / np.sqrt(m.BARS_PER_YEAR[interval])

    close_s = pd.Series(close)
    score_mom_log = np.log(close_s / close_s.shift(score_lookback)).to_numpy()
    score_local_vol = pd.Series(log_ret_padded).rolling(score_lookback).std().to_numpy()
    cmf_arr = _cmf_array(high, low, close, volume, score_lookback) if score_type == "cmf" else None
    hurst_arr = _hurst_like_array(close, hurst_window, hurst_q) if hurst_gate else None

    pos_dir_arr = np.zeros(n)
    strat_ret = np.zeros(n)
    pos_dir, stop_level, tp_level, pos_size, has_tp = 0, np.nan, np.nan, 0.0, False
    forced_dir = 0

    for t in range(1, n):
        prev_close = close[t - 1]
        if pos_dir == 0:
            if forced_dir != 0 and not np.isnan(atr[t]) and not np.isnan(realized_vol[t]) and realized_vol[t] > 0:
                pos_dir = forced_dir
                pos_size = min(target_vol_per_bar / realized_vol[t], m.LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr[t]
                has_tp = False
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * round_trip_cost / 2
            elif (not np.isnan(mom[t]) and mom[t] != 0 and not np.isnan(atr[t])
                    and not np.isnan(realized_vol[t]) and realized_vol[t] > 0):
                pos_dir = 1 if mom[t] > 0 else -1
                pos_size = min(target_vol_per_bar / realized_vol[t], m.LEVERAGE_CAP[coin])
                stop_level = prev_close - pos_dir * atr_mult * atr[t]
                tp_level = prev_close + pos_dir * tp_atr_mult * atr[t]
                has_tp = True
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close) - pos_size * round_trip_cost / 2
            forced_dir = 0
        else:
            stop_hit = (low[t] <= stop_level) if pos_dir == 1 else (high[t] >= stop_level)
            tp_hit = has_tp and ((high[t] >= tp_level) if pos_dir == 1 else (low[t] <= tp_level))
            if stop_hit or tp_hit:
                exit_px = stop_level if stop_hit else tp_level
                strat_ret[t] = pos_dir * pos_size * np.log(exit_px / prev_close) - pos_size * round_trip_cost / 2
                if tp_hit and not stop_hit:
                    raw_score = cmf_arr[t] if score_type == "cmf" else score_mom_log[t]
                    if np.isnan(raw_score):
                        cand_dir = 0
                    else:
                        if score_type == "vol_norm":
                            local_vol = score_local_vol[t]
                            chosen = raw_score / (local_vol * np.sqrt(score_lookback)) if local_vol > 0 else 0.0
                        else:  # raw or cmf are both already absolute-but-bounded readings
                            chosen = raw_score
                        signed_chosen = pos_dir * chosen
                        if signed_chosen > eps:
                            cand_dir = pos_dir
                        elif signed_chosen < -eps:
                            cand_dir = -pos_dir
                        else:
                            cand_dir = 0
                    if cand_dir != 0 and hurst_gate:
                        h = hurst_arr[t]
                        if np.isnan(h) or h <= hurst_threshold:
                            cand_dir = 0  # regime not confirmed trending -> flat instead
                    forced_dir = cand_dir
                else:
                    forced_dir = 0
                pos_dir, stop_level, tp_level, pos_size, has_tp = 0, np.nan, np.nan, 0.0, False
            else:
                strat_ret[t] = pos_dir * pos_size * np.log(close[t] / prev_close)
                if not np.isnan(atr[t]):
                    stop_level = max(stop_level, close[t] - atr_mult * atr[t]) if pos_dir == 1 else min(stop_level, close[t] + atr_mult * atr[t])
        pos_dir_arr[t] = pos_dir
    return pos_dir_arr, strat_ret


def _real_history_series_v2(coin: str, interval: str, lookback_bars: int, **kwargs) -> pd.Series:
    log_ret_hist, hi_hist, lo_hist, vol_hist = extract_market_tuples_v2(coin, interval)
    close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret_hist))])
    high_full = np.concatenate([[1.0], close_full[1:] * np.exp(hi_hist)])
    low_full = np.concatenate([[1.0], close_full[1:] * np.exp(lo_hist)])
    volume_full = np.concatenate([[vol_hist[0]], vol_hist])
    _, strat_ret = vol_target_returns_post_tp_v2(log_ret_hist, high_full, low_full, close_full, volume_full,
                                                  lookback_bars, ATR_K, coin=coin, interval=interval, **kwargs)
    df = load_candles_v2(coin, interval)
    return pd.Series(strat_ret, index=df.index).iloc[1:]


def _sharpe(s: pd.Series, bars_per_year: int) -> float:
    if len(s) < 2 or s.std() == 0:
        return np.nan
    return (s.mean() / s.std()) * np.sqrt(bars_per_year)


def _simulate_path_v2(coin, interval, lookback_bars, params, fixed_kwargs, n_bars,
                       log_ret_hist, hi_hist, lo_hist, vol_hist, rng, round_trip_cost=m.ROUND_TRIP_COST):
    block_len = BLOCK_LEN[interval]
    log_ret, high_frac, low_frac, volume = block_bootstrap_path_v2(log_ret_hist, hi_hist, lo_hist, vol_hist, n_bars, block_len, rng)
    close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret))])
    high_full = np.concatenate([[1.0], close_full[1:] * np.exp(high_frac)])
    low_full = np.concatenate([[1.0], close_full[1:] * np.exp(low_frac)])
    volume_full = np.concatenate([[volume[0]], volume])
    _, strat_ret = vol_target_returns_post_tp_v2(log_ret, high_full, low_full, close_full, volume_full,
                                                  lookback_bars, ATR_K, coin=coin, interval=interval,
                                                  round_trip_cost=round_trip_cost, **{**fixed_kwargs, **params})
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


def _run_barrier_sim_v2(coin, interval, lookback_bars, params, fixed_kwargs, bar_range=None,
                         n_paths=N_PATHS, horizon_days=HORIZON_DAYS, seed=0):
    rng = np.random.default_rng(seed)
    log_ret_hist, hi_hist, lo_hist, vol_hist = extract_market_tuples_v2(coin, interval, bar_range)
    n_bars = int(horizon_days * 24 / int(interval[:-1]))
    outcomes = {"pass": 0, "fail_static": 0, "fail_daily": 0, "unresolved": 0}
    for _ in range(n_paths):
        r = _simulate_path_v2(coin, interval, lookback_bars, params, fixed_kwargs, n_bars,
                               log_ret_hist, hi_hist, lo_hist, vol_hist, rng)
        outcomes[r] += 1
    return {k: v / n_paths for k, v in outcomes.items()}


def _outlier_check_v2(coin, interval, lookback_bars, params, fixed_kwargs, n_drop=20):
    s = _real_history_series_v2(coin, interval, lookback_bars, **{**fixed_kwargs, **params})
    bars_per_year = m.BARS_PER_YEAR[interval]
    full = _sharpe(s, bars_per_year)
    keep = s.abs().sort_values(ascending=False).index[n_drop:]
    dropped = _sharpe(s.loc[s.index.isin(keep)].sort_index(), bars_per_year)
    return full, dropped


def walk_forward_v2(coin: str, interval: str, lookback_bars: int, param_grid: list[dict], fixed_kwargs: dict) -> dict:
    bars_per_year = m.BARS_PER_YEAR[interval]
    series_by_param = {i: _real_history_series_v2(coin, interval, lookback_bars, **{**fixed_kwargs, **p})
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

    bar_range = (split, n_bars_series)
    sim = _run_barrier_sim_v2(coin, interval, lookback_bars, param_star, fixed_kwargs, bar_range=bar_range, n_paths=N_PATHS)

    return {"coin": coin, "interval": interval, "param_star": param_star,
            "train_sharpe_at_star": train_sharpe[i_star], "test_sharpe": test_sharpe,
            "test_max_dd": test_dd, "bar_range": bar_range,
            "oos_p_pass": sim["pass"], "oos_fail_static": sim["fail_static"],
            "oos_fail_daily": sim["fail_daily"], "oos_unresolved": sim["unresolved"]}


def _build_grid() -> list[dict]:
    grid = []
    for st in ("raw", "vol_norm"):
        for lbk in SCORE_LOOKBACK_GRID:
            for e in EPS_GRID[st]:
                for hg in (False, True):
                    grid.append({"score_type": st, "score_lookback": lbk, "eps": e, "hurst_gate": hg})
    for lbk in CMF_LOOKBACK_GRID:
        for e in CMF_EPS_GRID:
            for hg in (False, True):
                grid.append({"score_type": "cmf", "score_lookback": lbk, "eps": e, "hurst_gate": hg})
    return grid


def build_daily_series_v2(param_star_by_combo: dict) -> pd.DataFrame:
    cols = {}
    for (coin, interval), (lb, k) in LOCKED.items():
        bar_range, params = param_star_by_combo[(coin, interval)]
        s = _real_history_series_v2(coin, interval, lb, tp_atr_mult=TP_ATR_MULT_BY_COMBO[(coin, interval)], **params)
        test = s.iloc[bar_range[0]:bar_range[1]]
        daily = test.resample("1D").sum() * k * WEIGHT
        cols[f"{coin}_{interval}"] = daily
    return pd.DataFrame(cols).dropna(how="any")


if __name__ == "__main__":
    print("=== Sanity check: TP disabled (huge tp_atr_mult) must reproduce vol_target_returns(atr_mult=2.5), any score_type/hurst_gate ===")
    for (coin, interval), (lb, _k) in LOCKED.items():
        log_ret_hist, hi_hist, lo_hist, vol_hist = extract_market_tuples_v2(coin, interval)
        close_full = np.concatenate([[1.0], np.exp(np.cumsum(log_ret_hist))])
        high_full = np.concatenate([[1.0], close_full[1:] * np.exp(hi_hist)])
        low_full = np.concatenate([[1.0], close_full[1:] * np.exp(lo_hist)])
        volume_full = np.concatenate([[vol_hist[0]], vol_hist])
        _, ref = _baseline_vtr(log_ret_hist, high_full, low_full, close_full, lb, ATR_K, ATR_MULT_BASE, coin, interval)
        for st, hg in (("cmf", True), ("vol_norm", True)):
            _, post = vol_target_returns_post_tp_v2(log_ret_hist, high_full, low_full, close_full, volume_full, lb, ATR_K,
                                                     coin, interval, tp_atr_mult=1e6, score_type=st, score_lookback=5,
                                                     eps=0.0, hurst_gate=hg)
            match = np.allclose(ref, post, atol=1e-10)
            print(f"  {coin} {interval} score_type={st} hurst_gate={hg}: {'OK' if match else 'MISMATCH'}")
            if not match:
                sys.exit("TP-disabled baseline-reproduction sanity check failed.")

    print("\n=== H_POST_TP_SCORE_V2: per-combo walk-forward over score_type x lookback x eps x hurst_gate ===")
    grid = _build_grid()
    print(f"(grid size = {len(grid)} candidates per combo)")
    rows = []
    param_star_by_combo = {}
    for (coin, interval), (lb, _k) in LOCKED.items():
        tp = TP_ATR_MULT_BY_COMBO[(coin, interval)]
        wf = walk_forward_v2(coin, interval, lb, grid, fixed_kwargs={"tp_atr_mult": tp})
        full_sh, drop_sh = _outlier_check_v2(coin, interval, lb, wf["param_star"], {"tp_atr_mult": tp})
        rows.append({"coin": coin, "interval": interval, **wf, "full_sharpe": full_sh, "outlier_dropped_sharpe": drop_sh})
        param_star_by_combo[(coin, interval)] = (wf["bar_range"], wf["param_star"])

    df = pd.DataFrame(rows)
    fmt = {"train_sharpe_at_star": "{:.2f}".format, "test_sharpe": "{:.2f}".format,
           "test_max_dd": "{:.1%}".format, "oos_p_pass": "{:.1%}".format,
           "full_sharpe": "{:.2f}".format, "outlier_dropped_sharpe": "{:.2f}".format}
    print(df[["coin", "interval", "param_star", "train_sharpe_at_star", "test_sharpe",
              "test_max_dd", "oos_p_pass", "full_sharpe", "outlier_dropped_sharpe"]].to_string(index=False, formatters=fmt))

    print("\n=== Reference: corrected H_POST_TP_REENTRY (session 26 correction) and locked baseline ===")
    base = pd.read_json(Path(__file__).parent / "output" / "tsmom_walkforward.json")
    base = base[base.apply(lambda r: (r["coin"], r["interval"]) in LOCKED, axis=1)]
    print(base[["coin", "interval", "test_sharpe_at_lb_star", "test_max_dd", "oos_p_pass"]]
          .to_string(index=False, formatters={"test_sharpe_at_lb_star": "{:.2f}".format,
                                                "test_max_dd": "{:.1%}".format, "oos_p_pass": "{:.1%}".format}))

    print("\n=== Portfolio-level benchmark: pooled p_pass vs. locked baseline (81.0%/63.6%) and corrected H_POST_TP_REENTRY (82.2%/64.1%) ===")
    daily_df = build_daily_series_v2(param_star_by_combo)
    print(f"Common calendar overlap: {daily_df.index.min().date()} -> {daily_df.index.max().date()}, n={len(daily_df)} days")
    print("Pairwise correlation:")
    print(daily_df.corr().round(2).to_string())
    combined_hist = daily_df.sum(axis=1)
    pooled_sharpe = combined_hist.mean() / combined_hist.std() * np.sqrt(365) if combined_hist.std() > 0 else np.nan
    print(f"Pooled daily return: mean={combined_hist.mean():.5f}, std={combined_hist.std():.5f}, ann. Sharpe={pooled_sharpe:.2f}")

    daily_mat = daily_df.to_numpy()
    rng = np.random.default_rng(20260907)
    outcomes = [simulate_pooled_path(daily_mat, BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng) for _ in range(N_SIMS)]
    names = [o[0] for o in outcomes]
    for name in ["pass", "fail_static", "fail_daily", "unresolved"]:
        print(f"  {name:>12}: {names.count(name)/N_SIMS:.1%}")

    EPISODE = ("2026-08-17", "2026-08-27")
    mask = ~((daily_df.index >= EPISODE[0]) & (daily_df.index <= EPISODE[1]))
    ex = daily_df.loc[mask]
    ex_combined = ex.sum(axis=1)
    ex_sharpe = ex_combined.mean() / ex_combined.std() * np.sqrt(365) if ex_combined.std() > 0 else np.nan
    rng2 = np.random.default_rng(20260907)
    ex_outcomes = [simulate_pooled_path(ex.to_numpy(), BLOCK_LEN_DAYS, CYCLE_CAP_DAYS, rng2) for _ in range(N_SIMS)]
    ex_names = [o[0] for o in ex_outcomes]
    print(f"\nEx Aug 17-27 rally (n={len(ex)} days): ann. Sharpe={ex_sharpe:.2f}, "
          f"p_pass={ex_names.count('pass')/N_SIMS:.1%}, fail_static={ex_names.count('fail_static')/N_SIMS:.1%}")

    OUT.mkdir(exist_ok=True)
    df.drop(columns=["param_star"]).assign(param_star=df["param_star"].astype(str)).to_json(
        OUT / "tsmom_post_tp_score_v2.json", orient="records", indent=2)
