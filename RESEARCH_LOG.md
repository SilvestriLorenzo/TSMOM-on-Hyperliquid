# TSMOM research log

Curated, chronological, TSMOM-only excerpts from this project's full research journal
(the broader journal also covers ~15 other hypothesis threads — ETF flows,
microstructure, lead-lag, impulse-continuation, FX/commodities — most killed under the
same discipline applied here; see the main README for a short summary of the most
promising ones). Content below is copied verbatim from the original research log,
session by session, in the order it was actually written — including the walk-forward
corrections, a real look-ahead bug found and fixed, and every negative/mixed result
reported honestly alongside the positive ones.

---

### H_TSMOM — intraday vol-targeted trend-following with ATR trailing stop (session 10,
revisits H_A) — the most robust result in this project so far

User-proposed: vol-targeted time-series momentum on BTC/ETH/SOL at a 4-12h signal
horizon, sized via Magdon-Ismail & Atiya's expected-maximum-drawdown formula toward
~1.5% (half the Turbo static-drawdown budget), with Grossman-Zhou-style dynamic
de-risking as equity approaches the floor, citing arXiv:2602.11708 as a recent
application. Read that paper in full first: it does **not** actually use
Magdon-Ismail-Atiya or Grossman-Zhou — it's Bui & Nguyen's "AdaptiveTrend", a 150-coin
cross-sectional long/short framework (market-cap filtering, monthly Sharpe-based
selection, asymmetric 70/30 allocation) tested on Binance Futures H6 candles,
Sharpe 2.41. The honest comparison point for a simple BTC/ETH/SOL-only version is their
own **"Vol-Scaled TSMOM" benchmark row: Sharpe 1.83, MDD -16.1%** — not their
full-framework headline, which depends on components this hypothesis doesn't include.
Two things from the paper *do* transfer directly: their own timeframe sweep peaks at H6
(bracketed here, since Hyperliquid has no native 6h interval — 422 on that string — by
4h/8h/12h instead), and their ablation shows the ATR-based dynamic trailing stop is by
far the single biggest contributor (Sharpe 2.41→1.68, MDD -12.7%→-22.4% without it).

**Data**: native Hyperliquid candles fetched directly (`research/fetch_tsmom_candles.py`)
at 4h/8h/12h — same 5000-candle-per-request cap as `fetch_reference_data.py`, but a
coarser interval buys much more real history per request: 4h → 833 days (2024-05-20
onward), 8h → 1537 days (2022-06-16 onward), 12h → 1704 days (2022-01-01 onward), all
three coins.

**Naive signal (sign of the trailing 1-3 bar return, position re-flipped every bar, no
trailing stop) — killed, consistent with H_A's earlier finding.** Across all 27
(coin × interval × lookback) combinations: mostly negative Sharpe (BTC 4h: -2.98 to
-1.32), a few small positive ones scattered inconsistently (SOL 12h lookback=1: +0.48,
lookback=2: +0.02, lookback=3: +0.91 — a non-monotonic zigzag across adjacent parameter
values, the signature of noise, not signal). This independently reproduces H_A's
daily-horizon kill at intraday resolution: short-horizon crypto returns don't show
naive sign-continuation, consistent with the well-documented tendency toward
short-horizon mean-reversion rather than momentum.

**Adding the ATR trailing-stop exit (Bui & Nguyen's own biggest contributor,
α=2.5, ATR-14, both taken directly from their reported optimum rather than tuned on our
data) changes the picture entirely** — but the first attempt at this had a real bug
worth documenting: checking the stop against that same bar's own close/ATR (only known
*at* that bar's close) and crediting the bar's return using the already-exited position
silently let the backtest dodge the entire adverse bar an ATR stop exists to catch,
rather than capturing the loss actually taken up to the stop price. This inflated
Sharpe to an implausible 2.2-3.6 uniformly across the grid, with results nearly
identical across different lookback values (a second tell — the entry signal was
barely mattering once look-ahead let the exit rescue every trade). Fixed by fixing the
stop level using only data through bar *t-1*, checking it against bar *t*'s actual
intrabar low/high (real OHLC data), and truncating the captured return to the stop
price on breach rather than the full bar.

**Corrected results: uniformly positive across all 27 combinations (Sharpe 0.41-1.24)**
— every single one, where the naive version was mostly negative. Best three: BTC 4h
lb=1 (Sharpe 1.24, MDD -16.9%), ETH 12h lb=1 (1.22, -22.2%), SOL 12h lb=2 (1.22, -14.0%).
All below the paper's own Vol-Scaled TSMOM reference (1.83) but real, and — critically —
**far more robust to outlier removal than H_ETF ever was**: dropping the 20
largest-magnitude bars (of 3400-5000) degrades BTC 4h from 1.24→0.26, barely moves ETH
12h (1.22→0.69), and *improves* SOL 12h (1.22→1.07). Compare to H_ETF, where dropping 5
of 495 days collapsed its t-stat by 59%. This is the expected signature of genuine
trend-following (a systematic rule mechanically capturing large moves, not a fragile
dependence on one or two lucky episodes) — the healthiest robustness profile of
anything tested in this project.

**Leverage-matched buy-and-hold control** (same discipline as H_ETF's same-window
baseline): confirms this isn't just "being long a bull market" — a buy-and-hold
position at the strategy's own average leverage had *worse* Sharpe (often negative:
-0.31 to +0.69) and dramatically worse max drawdown (-34% to -55% vs the strategy's
-8% to -28%) over the identical windows, including windows where buy-and-hold lost
money outright while the strategy made large gains (ETH 4h: BH -13.7% total vs strategy
+54% annualized).

**Translated into an actual Propr `p_pass` (`research/tsmom_barrier_sim.py`),
Magdon-Ismail & Atiya-calibrated sizing built and tested, Grossman-Zhou-style de-risking
built and tested — both add-ons made things worse, not better.** Full path-dependent
Monte Carlo, block-bootstrapping the raw (return, high, low) market-data tuples in
contiguous blocks (preserving local trend/vol clustering, matching arXiv:2602.11708's
own significance-testing methodology) and re-running the entire entry+ATR-stop signal
fresh on each synthetic path, checking Propr's two barriers directly against real
bar-level granularity (no Brownian-bridge approximation needed, unlike H_ETF/H_STRUCT).

1. **Baseline (the fixed-vol-target sizing already tested above, no MI&A, no GZ),
   400 Monte Carlo paths per combo: `p_pass` = 24.5% (BTC 4h), 27.5% (ETH 12h), 28.5%
   (SOL 12h)** — in the same range as H_ETF's 28.4%, but built on a materially more
   outlier-robust backtest. Failures are mostly the static-drawdown barrier (63-67%),
   a smaller share the daily-loss barrier (7.5-9.25%), zero unresolved.
2. **Magdon-Ismail & Atiya's own formula was verified against the paper's published
   reference table** (Appendix B of Magdon-Ismail, Atiya, Pratap & Abu-Mostafa, 2004, J.
   Applied Probability 41:147-161 — full text saved locally,
   `research/output/papers/magdon_ismail_atiya_2004.pdf`) before use
   (`research/mi_atiya_drawdown.py`). Useful simplification found along the way: since
   E[max drawdown] scales *exactly linearly* with a uniform leverage multiplier (the
   dimensionless parameter the whole formula depends on, alpha^2 = mu^2*T/2*sigma^2, is
   leverage-invariant — scaling mu and sigma by the same k leaves their ratio, and
   therefore alpha, unchanged), calibrating "the leverage that targets a given expected
   drawdown" needs no root-finding, just one evaluation at unit leverage and a linear
   scale.
3. **Applied literally (1.5% target, the user's own "half the budget" framing) at any
   horizon from 2 weeks to 3 years, the calibrated leverage is too small for the
   strategy to do anything.** Even at a 14-day horizon — far shorter than this project's
   standard 1095-day convention — the already-validated fixed-vol leverage's *own*
   implied E[MDD] is already 4.6-4.7%, above the 1.5% target; hitting 1.5% requires
   cutting leverage to 0.02-0.14x (vs. the validated ~0.2-0.45x), and at the standard
   1095-day horizon that falls to 0.012-0.021x. Barrier-sim result at that horizon:
   93.3-95.5% of paths **unresolved** (neither pass nor fail — equity barely moves),
   pass rate ~0%. This isn't a bug: it's the correct, if unhelpful, consequence of
   asking for a drawdown budget smaller than the strategy's natural volatility can
   support over any horizon long enough to matter.
4. **The Grossman-Zhou-style overlay (cushion-proportional exposure, `size = leverage *
   max(0, (equity - floor) / equity)`, layered on top of the already-validated fixed-vol
   leverage rather than the MI&A one, to isolate this effect specifically) is worse, not
   better: 100% of paths unresolved, zero pass, zero fail**, confirmed at 400 paths per
   combo. Propr's static floor is fixed (not a ratcheting high-water mark, unlike the
   framework Grossman-Zhou was built for), so a naive cushion-proportional rule reduces
   exposure the moment equity dips even slightly below the starting balance (which
   happens immediately from costs and the first losing trades) — and once exposure is
   small, the strategy can't generate enough return to ever reach +9%, while also being
   too de-risked to breach the floor. The account gets stuck in permanent limbo rather
   than genuinely protected. A less aggressive multiplier than the m=1 used here might
   avoid this, but wasn't tested — this specific, literal implementation of the user's
   proposal actively hurts rather than helps.
5. **Bottom line: the raw signal's own simple fixed-vol-target sizing — already built,
   already outlier-checked — is the best-performing configuration found so far.** Both
   halves of the more sophisticated proposed sizing framework, implemented as specified,
   make things worse. If this is pursued further, the productive next step is tuning the
   CPPI multiplier and/or the MI&A target drawdown away from the literal "half the
   budget" framing, not abandoning the underlying signal.
6. Each asset was tested independently — no cross-asset portfolio combination (equal
   weight or otherwise) tried yet.
7. ATR parameters were taken directly from the cited paper's reported optimum, not
   calibrated on this data — deliberately, to avoid in-sample overfitting, but this
   also means they may not be the best fit for Hyperliquid specifically.

**Walk-forward validation (session 11, `research/tsmom_walkforward.py`) — the edge and
the combo ranking both hold up out-of-sample, with a real twist on which combos lead.**
Closes the gap flagged above (item 6's underlying issue, generalized): session 10 picked
`lookback_bars` and which 3 of 9 (coin, interval) combos to headline using the same
full-history window the numbers were reported on. Fixed with a genuine chronological
70/30 train/test split, all 9 combos (not just the original top 3, to avoid moving the
selection bias up one level): select `lookback_bars` by Sharpe on train only, report
Sharpe/MDD/`p_pass` on the untouched test slice, `p_pass` computed via
`tsmom_barrier_sim.run_barrier_sim` with its block-bootstrap source and fixed-vol
leverage calibration both restricted to the same test bars (new `bar_range` parameter,
backward-compatible — verified the existing `__main__` demo reproduces session 10's
exact numbers unchanged before trusting the extension).

| Coin | Interval | lb* (train) | Train Sharpe | Test Sharpe @ lb* | Test max DD | OOS `p_pass` |
|---|---|---|---|---|---|---|
| BTC | 4h | 1 | 0.89 | **1.95** | -11.0% | **30.5%** |
| BTC | 8h | 1 | 0.60 | 0.72 | -24.1% | 27.5% |
| BTC | 12h | 1 | 1.18 | 0.57 | -28.5% | 25.2% |
| ETH | 4h | 1 | 1.03 | 0.61 | -23.1% | 28.2% |
| ETH | 8h | 1 | 0.97 | 1.54 | -15.2% | **35.2%** |
| ETH | 12h | 1 | 1.09 | 1.54 | -13.6% | 26.2% |
| SOL | 4h | 3 | 0.71 | 1.45 | -16.3% | 27.8% |
| SOL | 8h | 1 | 0.82 | 1.26 | -15.1% | 28.0% |
| SOL | 12h | 2 | 1.18 | 1.32 | -12.5% | 25.8% |

**Every one of the 9 combos has a positive, cost-surviving out-of-sample Sharpe (0.57 to
1.95) and an OOS `p_pass` of 25.2-35.2%** — at or above session 10's in-sample headline
numbers (24.5%/27.5%/28.5% for BTC 4h/ETH 12h/SOL 12h specifically), not below them. The
aggregate selection-artifact gap (mean train Sharpe minus mean test Sharpe at the
train-selected lookback) is **-0.28 — negative**, meaning test-period performance was on
average *better* than train, the opposite of the shrinkage an overfit combo-selection
would produce. Spot-checked one combo (BTC 4h) by hand against a from-scratch Sharpe
recomputation and confirmed the train/test bar-index alignment between the return-series
slicing and the barrier-sim's `bar_range` matches exactly (max diff ~1e-18, floating-point
noise). **This is now the most walk-forward-robust result in the project** — unlike
H_ETF, which lost most of its significance in its own walk-forward correction, H_TSMOM's
edge does not shrink out-of-sample.

**One real reshuffle worth flagging**: the previous "top 3" (BTC 4h, ETH 12h, SOL 12h)
are no longer clearly the best out-of-sample — ETH 8h now shows the single highest OOS
`p_pass` (35.2%), and lookback selection landed on lb=1 for 7 of 9 combos (SOL 4h and SOL
12h are the exceptions). Treat ETH 8h as a live candidate alongside the original three,
not a replacement — 9 combos is still a small sample to pick a new "best of" from, and
this reshuffle is itself evidence that any single-window ranking (including this one)
carries some noise.

**Sizing-overlay retune (session 11, `research/tsmom_sizing_retune.py`), built on the
walk-forward-honest baseline above — the MI&A half of the user's original proposal
genuinely helps once freed from the literal "half the budget, 3-year horizon" framing;
the CPPI de-risking half still mostly doesn't.**

1. **MI&A target_mdd × calibration-horizon sweep** (target ∈ {1.5%, 2.25%, 3%, 4.5%},
   horizon ∈ {30, 90, 180, 365} days, vs. session 10's fixed 1.5%/1095d): swept for each
   of the 9 walk-forward-validated combos, evaluated on the same held-out test-period
   bootstrap source the walk-forward baseline uses. **Every single combo has a
   configuration that beats its own honest fixed-vol baseline** — by +2pp (SOL 8h) to
   +13pp (BTC 4h: 30.5%→43.5%). This is not one lucky cell: on average **53% of the
   16-cell grid beats baseline per combo** (BTC/ETH: 44-75%; SOL weaker at 19-63%,
   SOL 8h the softest at 19%). The mechanism is real and makes sense in hindsight: the
   fixed-20%-annualized-vol-target leverage was calibrated to match a *reference paper's*
   vol level, with no reference to Propr's actual drawdown budget at all — it was never
   claimed to be `p_pass`-optimal, just the first thing tested. MI&A, once given a
   target/horizon that actually matches the strategy's real pace (best cells cluster at
   30-365 day horizons and 2.25-4.5% targets — i.e. close to or somewhat above the full
   3% budget, not half of it, and weeks-to-months, not 3 years) explicitly reasons about
   the drawdown-vs-target tradeoff and finds a **generally lower** leverage than the
   fixed-vol convention that produces both fewer failures and more passes at once (e.g.
   BTC 4h's best cell uses leverage 0.130 vs. the fixed-vol baseline's 0.472, yet raises
   `p_pass` from 30.5% to 43.5% while *lowering* the failure rate).
2. **Caveat, applied with the same discipline as every sweep in this project**: picking
   the best of a 4×4=16-cell grid per combo, scored on the same single held-out test
   window the walk-forward baseline itself was validated on, is itself a form of
   in-sample selection — there is no third, still-unseen data slice left to validate the
   *specific* winning cell against. The qualitative finding (properly-horizon/target-
   calibrated MI&A sizing beats the naive fixed-vol convention, broadly across the grid,
   not just at one point) is well-supported by the 53%-of-grid robustness check above.
   The exact reported uplift for any single "best" cell should be treated the way this
   project treats every other cherry-picked-best number (the original 27-combo sweep,
   the top-3 headline) — probably real in direction, likely optimistic in magnitude.
3. **Budget-normalized CPPI de-risking** (`cppi_budget_c` sweep ∈ {0.1, 0.25, 0.5, 1.0}):
   a genuine bug in session 10's literal formula was found in the course of
   generalizing it — it normalized the equity cushion against *current equity* (~1.0),
   not against the actual 3%-of-equity drawdown *budget* it was meant to protect, an
   implicit ~33x-too-conservative ratio baked into the formula itself (independent of
   any multiplier), which alone explains why every path went to "unresolved." The
   corrected, budget-normalized version is far less broken but still **does not clearly
   help**: only 3 of 9 combos (BTC 8h, ETH 12h, SOL 4h) find a `c` that beats the honest
   baseline, and the winning `c` is usually 1.0 (the *least* aggressive de-risking
   tested, i.e. closest to no overlay at all) — consistent with session 10's original
   conclusion that de-risking hurts this strategy specifically, because Propr's floor is
   fixed rather than a ratcheting high-water mark, so pulling exposure down near the
   floor mostly just delays recovery rather than protecting anything.
4. **Bottom line, updated from session 10**: the sizing overlay is worth pursuing
   further, but only the MI&A half, and only once decoupled from the literal "half the
   3-year budget" framing the user originally proposed — that framing was the problem,
   not the underlying MI&A machinery, which (correctly calibrated) is now the best
   sizing found for this strategy, ahead of both the plain fixed-vol baseline and the
   CPPI-style overlay in either its literal or budget-normalized form.

Still open: funding costs (not yet modeled, non-trivial given ~10-bar average holds) and
a cross-asset portfolio combination (BTC/ETH/SOL together) — both explicitly deferred to
a future session, not part of this one.

**L2-cost validation (session 12, `research/tsmom_l2_cost_check.py`) — the flat 9bps
assumption holds up; real spread/depth cost changes nothing.** Every H_TSMOM backtest so
far used a flat 9bps round-trip cost meant to represent core-dex's taker fee alone
(4.5bps/side × 2, `PROPR.md` §7), with no spread/depth cost included — session 6/9's
H_COST work already showed this is a separate, additive cost, small for BTC/ETH but
never checked against SOL or against TSMOM's own turnover specifically (its ~10-bar
average hold means far more round trips per year — 58-216/year across the 9 combos —
than the buy-and-hold-style checks H_COST originally ran).

Reused `l2_cost_analysis.py`'s exact vectorized walk-the-book logic against the
collector's L2 book sample (~34-35K matched bid/ask snapshots per coin, continuous
collection since session 6) at notionals from $1K-$50K, spanning the leverage values
found in the walk-forward/sizing-retune work (~0.02-0.5x of equity):

| Coin | Spread+depth cost @ $10K (p50/p90, bps) | TOTAL real cost (+9bps fee) | Flat assumption |
|---|---|---|---|
| BTC | 0.13 / 0.82 | 9.13 / 9.82 | 9.0 |
| ETH | 0.41 / 0.82 | 9.41 / 9.82 | 9.0 |
| SOL | 0.97 / 1.93 | 9.97 / 10.93 | 9.0 |

BTC/ETH confirm session 6/9's finding (negligible spread on top of fees). SOL is
noticeably wider — real total cost 11-32% above the flat assumption at the tail (up to
+2.9bps at $50K/p90) — the first time in this project SOL specifically has been checked
against real book depth for this strategy.

**Threaded the corrected per-coin cost through the actual backtest (not an estimate):**
extended `tsmom_barrier_sim.py`'s cost handling from a hardcoded module constant to a
`round_trip_cost` parameter (backward-compatible default = the original flat 9bps,
verified the existing `__main__` demo reproduces identical numbers before trusting the
change), threaded through `tsmom_walkforward.py`, and re-ran all 9 walk-forward combos
at both the $10K p50 and p90 real-cost figures:

| Coin | Interval | Flat `p_pass` | @ real p50 cost | @ real p90 cost (stress) |
|---|---|---|---|---|
| BTC | 4h | 30.5% | 30.5% | 30.5% |
| BTC | 8h | 27.5% | 27.3% | 27.3% |
| BTC | 12h | 25.2% | 25.2% | 25.2% |
| ETH | 4h | 28.2% | 28.2% | 28.2% |
| ETH | 8h | 35.2% | 35.2% | 35.2% |
| ETH | 12h | 26.2% | 26.2% | 26.2% |
| SOL | 4h | 27.8% | 27.5% | 27.3% |
| SOL | 8h | 28.0% | 27.8% | 27.8% |
| SOL | 12h | 25.8% | 25.8% | 25.5% |

**Verdict: no change to anything.** The largest movement anywhere, including SOL's
wider real spread and the deliberately pessimistic p90-stress cost, is -0.5pp. This
closes the last unchecked cost-model gap for H_TSMOM cheaply and definitively — the flat
9bps convention used throughout this hypothesis was never a material risk to any
reported number, for any of the three coins.

**CORRECTION (session 14) — the Monte Carlo underlying every session 10-13 `p_pass`
number tested the wrong sizing rule.** While pinning down a deployment config, found
that `tsmom_barrier_sim.py`'s Monte Carlo sized every synthetic path with a CONSTANT
leverage multiplier — no per-bar vol adjustment — adopted in session 10 so Magdon-Ismail
& Atiya's closed-form (which assumes constant drift/vol) applied directly. That was
quietly a different, weaker strategy than the one actually validated: on BTC 4h's
walk-forward test slice, the real per-bar vol-targeted process (what
`tsmom_hypothesis.py`'s backtest_trailing_stop actually tested, and what every Sharpe
number in this document refers to) has Sharpe **1.95**; the constant-leverage version of
the identical entry/exit signal, same window, has Sharpe **1.43** — a 37% gap, confirmed
directly, not estimated. Vol-targeting is doing real work here (a standard
trend-following effect — sizing down before volatile trades and up before quiet ones
improves the signal-to-noise ratio of the trade sequence), and every `p_pass` figure in
sessions 10-13 was computed against the inferior constant-leverage proxy instead.

**Fixed**: `simulate_path` now runs genuine per-bar vol-targeted sizing on every
synthetic path (position size = min(target_vol_per_bar / trailing 20-bar realized vol,
leverage cap), fixed at entry — exactly `tsmom_hypothesis.py`'s rule), with `leverage`
repurposed as a scalar multiplier k applied on top (k=1.0 = the plain validated strategy,
no overlay; MI&A/CPPI scale k from there) — also a more faithful reading of the user's
original session-10 proposal (MI&A sizing a vol-targeted TSMOM, not replacing its
vol-targeting). `calibrate_mia_leverage` now estimates mu/sigma from this real process,
not the old constant-size stand-in. **Re-ran the full walk-forward with the corrected
mechanics — the effect is real but not uniform in direction**, unlike a simple "the old
numbers were all too conservative" story:

| Coin | Interval | Old (flawed) `p_pass` | Corrected `p_pass` | Direction |
|---|---|---|---|---|
| BTC | 4h | 30.5% | 24.2% | worse |
| BTC | 8h | 27.5% | 25.5% | worse |
| BTC | 12h | 25.2% | 23.5% | worse |
| ETH | 4h | 28.2% | 24.8% | worse |
| ETH | 8h | 35.2% | 29.2% | worse |
| ETH | 12h | 26.2% | 31.2% | **better** |
| SOL | 4h | 27.8% | 25.8% | worse |
| SOL | 8h | 28.0% | 25.8% | worse |
| SOL | 12h | 25.8% | 24.8% | worse |

Most combos got worse, one (ETH 12h) got better. The mechanism, visible directly in the
failure-mode breakdown: genuine vol-targeting sizes up aggressively after a calm period
(potentially near the leverage cap) and that position is now oversized if volatility
jumps right after entry — a real, well-documented procyclical-exposure risk of
trailing-vol sizing. BTC 4h's daily-loss failure rate rose from 4.0% to 10.8% under the
correction; the static-drawdown rate barely moved — consistent with short, sharp
post-entry vol spikes being the specific new failure mode the constant-leverage
approximation had been hiding. **Net effect: H_TSMOM's edge is still real (every combo
still clears its cost floor with a positive, walk-forward-validated Sharpe — that part
of the walk-forward work is untouched by this bug, since Sharpe was always computed on
the correct vol-targeted series), but its `p_pass` is on average a few points lower than
previously reported, and the previous coin/interval ranking partially reshuffles** (ETH
12h now leads the original three; ETH 8h remains the single best combo overall). The
sizing-retune sweep was also rerun end to end — the qualitative finding survives
unchanged: every one of the 9 combos still finds an MI&A configuration that beats its
(corrected) baseline, and the budget-normalized CPPI overlay still mostly doesn't help
(2 of 9 beat baseline, same as before).


---

### H_TSMOM deployment config, locked (session 13/14)

Purpose: turn the research above into ONE fixed, written-down config per candidate
combo — not a number re-optimized after the fact. Two sizing decisions were
deliberately made WITHOUT searching the sizing-retune grid for the best cell (that
grid's argmax is scored on the same held-out test window the walk-forward baseline
itself was validated on — picking a fresh argmax "for real this time" would just repeat
the same in-sample-selection mistake one more time):

1. **MI&A target drawdown: fixed at 3%**, Propr's own full static-drawdown budget
   (`PROPR.md` §4) — a number read directly off the rulebook, not fit to data.
2. **MI&A calibration horizon: fixed at the strategy's own measured median
   bars-to-resolution** under the plain (k=1) vol-targeted baseline — no sizing search
   at all, just observing how fast the strategy naturally resolves (pass or fail,
   whichever comes first) and converting to days. This replaces both session 10's
   arbitrary 1095-day convention (shown to be far too long — leverage calibrated to it
   was too small to do anything) and the sizing-retune grid's swept horizons (each cell
   an arbitrary guess) with a number measured from the strategy's own behavior.

Candidates: the four combos strongest on out-of-sample `p_pass`, parameter stability
(train-selected lookback matching the test-optimal one), and MI&A-grid robustness (how
much of the 16-cell sweep beat baseline, not just the single best cell) — BTC 4h, ETH
8h, ETH 12h, SOL 12h (SOL 12h was one of session 10's original three, kept for an
explicit side-by-side despite scoring weakest on the third criterion).

| Coin | Interval | lb* | Baseline `p_pass` (k=1) | Measured horizon | Locked config `p_pass` | Leverage k | For reference: sizing-grid argmax |
|---|---|---|---|---|---|---|---|
| BTC | 4h | 1 | 24.2% | 16 days | **32.4%** | 0.479 | 38.0% |
| **ETH** | **8h** | **1** | **29.2%** | **22 days** | **36.0%** | **0.488** | 46.2% |
| ETH | 12h | 1 | 31.2% | 24 days | 33.0% | 0.479 | 37.5% |
| SOL | 12h | 2 | 24.8% | 26 days | 29.2% | 0.437 | 27.5% |

All four candidates gain 4.6-8.2pp over the plain (no-overlay) baseline from this
principled, non-searched choice — most of the sizing grid's demonstrated improvement is
captured without cherry-picking a specific cell (the gap to the grid's argmax is 4-10pp,
the honest cost of not searching). **ETH 8h leads** on the locked config (36.0%,
`p_pass`), consistent with its lead on every other metric computed this session.

**This is the config to carry into Phase 4 (paper/shadow validation)**: signal = sign of
the 1-bar (BTC 4h, ETH 8h, ETH 12h) or 2-bar (SOL 12h) trailing return, ATR-14 trailing
stop at 2.5x (Bui & Nguyen's reported optimum, not tuned on this data), position size =
`min(target_vol_per_bar / trailing-20-bar realized vol, per-asset leverage cap) × 0.479`
(0.488 for ETH 8h, 0.437 for SOL 12h) fixed at entry, held through the trade. Real L2
cost (session 12) confirmed the flat 9bps convention used in all these numbers is not a
material risk. Not yet decided: which of the four (or how many, staggered) to actually
run as paid challenge attempts, and the $5K/$10K/$100K account-size question — both
orthogonal business decisions, not technical ones this analysis can resolve alone.

**Correction (session 15): "Measured horizon" column above is at k=1, not at the locked
leverage.** It was only ever meant as the *input* to MI&A's calibration (how fast the
unscaled baseline resolves), not a claim about the locked config's own pace. Recomputed
directly at each combo's actual locked k, median days-to-resolution is roughly 3x longer
(k<1 slows the walk toward both barriers): BTC 4h 59d, ETH 8h 63d, ETH 12h 70d, SOL 12h
80d. Use these, not the table above, for any "how long until this resolves" question.


---

### H_TSMOM funded-phase economics (session 15)

Deployment-config above answers "does one evaluation pass, fail, or stall" — this
answers "if it passes, and I keep going, how much per month." New file:
`research/tsmom_funded_phase_sim.py`, a renewal simulation on top of the exact same
`simulate_path` (same locked k, same walk-forward test-slice bootstrap source): policy
is sweep immediately after every +9% pass (banks it, resets that account to its starting
balance, replays an identical fresh cycle at the same k — no re-purchase, since sweeping
doesn't revoke the funded account per `PROPR.md` §4/§8), and a fail (-3% static or
daily) is modeled as losing the funded slot, requiring a fresh $125 (25K-tier) evaluation
purchase to re-enter.

**Flagged modeling assumption, not confirmed in `PROPR.md`**: whether the evaluation's
own +9% gain becomes the first sweepable payout (this sim's assumption, and the standard
reading of "1-Step" instant-funding-style challenges), or whether a second +9% cycle
inside the funded account is required first. Verify against Propr's actual dashboard/
T&Cs before relying on this financially — if the latter, treat every number below as
roughly one extra cycle's delay from correct, not fundamentally different.

1000 runs × 3 years, $25K account:

| Coin | Interval | $/mo mean | $/mo median | p5-p95 | P(net<0 over 3y) | passes/yr | fails/yr |
|---|---|---|---|---|---|---|---|
| BTC | 4h | $195 | $200 | $52-$334 | 1.0% | 1.50 | 2.88 |
| **ETH** | **8h** | **$194** | **$195** | **$64-$328** | **0.5%** | 1.46 | 2.32 |
| ETH | 12h | $134 | $126 | $18-$259 | 1.7% | 1.07 | 2.54 |
| SOL | 12h | $106 | $105 | $10-$221 | 3.7% | 0.85 | 2.13 |

p_pass, cycle length, and the 80%/9% payout math are all size-independent, and the
challenge fee is ~0.45-0.50% of size at every tier — so this scales close to linearly
with capital deployed (×2 for $50K, ×4 for $100K, plus a small further gain from the
$100K tier's better fee-per-dollar). P(net<0) being low at every combo, over a 3-year
simulated horizon, reflects that failed-cycle fee cost is a small drag relative to
steady banked gains at these de-risked leverages — not a claim about the strategy's live
edge holding up, which is a separate, unresolved question (below).

**Decay risk, honestly assessed, not modeled**: time-series momentum is one of the
longer-documented systematic risk premia in the literature (Moskowitz/Ooi/Pedersen 2012,
and follow-on work showing multi-decade, multi-asset-class persistence) — it isn't a
microstructure inefficiency that typically vanishes in weeks. But this project's own
out-of-sample validation window is short (the walk-forward test slices are only
~14-17 months of crypto history, whatever regime that happened to be), crypto market
structure is evolving fast in both directions (more systematic/CTA participation could
crowd this exact signal, or strengthen it via more momentum-chasing flow), and nothing
here constitutes a live-decay check. The honest mitigation is empirical monitoring, not
a confident prediction either way: Phase 4 should track realized live/paper Sharpe
against the backtest's, on a rolling window, with an explicit pre-committed kill
criterion (e.g., pause and re-validate if trailing-90-day Sharpe falls persistently
below some fraction of the backtest figure) — this is on the Phase 4 punch list.


---

### H_TSMOM_FAST — the production TSMOM signal itself, run at 1m/5m instead of 4h/8h/12h (session 18)

Different question from H_IMPULSE above (a new mechanic, ported from elsewhere): this
asks whether the exact strategy already running live in Beta (`tsmom_beta_live.py`) —
sign of the most recent bar's return, vol-targeted sizing to a 20% annual target,
leverage-capped per PROPR.md, optional ATR trailing-stop exit — still has an edge if you
just point its own unmodified signal/sizing/exit logic (`research/tsmom_hypothesis.py`,
untouched) at a faster bar. Same 1m/5m data and retention caveat as H_IMPULSE (~3.6
days / ~17.5 days, thin single window).

**Result: uniformly, catastrophically negative — every coin, every lookback (1-30 bars),
both variants (naive re-flip and ATR trailing stop), both intervals.** Sharpes range
from roughly -9 to -540; total returns over the sample range from -10% to -91% (in a
matter of *days*, not years). Best-by-Sharpe cell in every grid is still deeply
negative — there is no lookback/interval combination that survives.

**Why, mechanically (no ambiguity here, unlike H_IMPULSE's noisier picture)**: two
compounding effects. (1) H_IMPULSE already showed 1-minute BTC/ETH/SOL returns are
reversal-dominated, not trend-following — the same sign flip that kills a
continuation bet also kills a lookback-1-bar momentum signal that re-enters on the same
noisy sign. (2) At 1m bars with no threshold filter, the signal re-flips direction most
bars, so turnover approaches 100%/bar — at the standard 9bps round-trip cost convention
that alone is ~4.5bps of cost *per bar*, compounding over thousands of bars/day. Unlike
H_TSMOM's real 4h/8h/12h edge (Sharpe ~1.4-2, H_COST-confirmed low real cost), there is
no bar-count-per-year regime here where a naive re-flip signal is cheap enough to survive
its own turnover.

**Verdict: correctly set aside.** Confirms H_TSMOM's own choice of a 4-12h signal
horizon (locked session 13/14) wasn't an arbitrary starting point — going faster with the
same mechanism actively destroys the edge rather than just diluting it. Not worth
retrying with a threshold filter or lower assumed cost: the reversal effect (1) is
structural to this frequency regardless of cost, and would need a fundamentally
different signal (e.g. explicitly betting reversal, still bounded by H_IMPULSE's finding
that reversal's raw magnitude is smaller than round-trip cost) to have any chance.


---

### H_TSMOM_GAP, H_OFI, H_FUNDCYCLE — three "faster than H_TSMOM" candidates (session 20, user-proposed)

Per user request after H_LEADLAG's kill, scoped explicitly to "something faster than the
current [4-12h] strategy" — three mechanistically distinct candidates, each cheap to test
against data already on hand or one quick fetch away.

**H_TSMOM_GAP — the untested 15m-2h band between the killed 1m/5m (H_TSMOM_FAST, session
18) and the locked 4h/8h/12h.** Reuses `tsmom_hypothesis.py`'s exact validated
backtest/backtest_trailing_stop functions unmodified, just extending the interval grid
(`research/tsmom_gap_hypothesis.py`).

- **15m: clean, decisive kill** — both the naive and ATR-stop variants are uniformly
  strongly negative (naive Sharpe -25 to -63; ATR-stop -2.4 to -5.4), and a genuine
  train/test walk-forward split confirms it (test Sharpe -2.99 to -9.68 for all three
  coins). The 1m/5m reversal-dominance finding extends cleanly up through 15m — no
  ambiguity here.
- **30m-2h: not confirmed, and a real methodological catch worth flagging.** The
  in-sample ATR-stop grid shows a striking coin split — BTC/SOL mostly negative through
  1h (only marginal at 2h, Sharpe ~0.5-0.8), while ETH is positive from 30m onward
  (Sharpe 1.1-1.75, comparable to the locked 4-12h combos' own OOS numbers). But a
  walk-forward check (lb selected on a 70% train slice, Sharpe-only — the full
  barrier-sim p_pass machinery wasn't extended to this interval band; its
  BARS_PER_DAY/BLOCK_LEN tables are hardcoded to 4h/8h/12h in `tsmom_barrier_sim.py`, not
  worth generalizing before a Sharpe-only gate says it's worth the effort) shows most
  cells **flip sign between train and test** (BTC and SOL especially: train Sharpe
  negative or near-zero, test Sharpe +2 to +4) — the classic regime-dependence signature,
  not a found edge. Checked directly and confirmed: **both ETH's 1h and 2h test windows
  (2026-07-06 onward and 2026-05-04 onward) contain the exact 2026-08-17/08-27 rally
  episode already flagged in H_ETF's outlier diagnosis as the dominant driver of *that*
  result.** The same two-week window is very plausibly inflating multiple, superficially
  independent-looking tests across this project — a genuinely useful catch (a shared
  confound across hypotheses, not just within one), not evidence of a real 30m-2h ETH
  edge. **Follow-up, same session, resolves this decisively rather than leaving it
  open**: tried to get a longer, cleaner history via a paginated fetch
  (`fetch_candles_paginated` in `fetch_reference_data.py`), which surfaced a real,
  previously-mis-documented fact about the data source itself — the "5000 candles per
  request" cap this project's code has assumed since session 3 is wrong. Verified
  directly (a 1h request for a 30-day window ~500 days ago returns 0 candles, the
  identical window at 1d resolution returns 31): it's a **server-side retention limit**
  on sub-daily intervals, not a per-call size cap, so pagination cannot reach data that
  was never kept — 30m/1h/2h history is structurally capped at ~104/208/417 days from
  Hyperliquid's public API, permanently, not just today. With more history off the
  table, ran the direct outlier-excision check instead (same method as H_ETF's original
  diagnosis): computed the full return series for every coin/interval/lookback in the
  30m-2h band and compared Sharpe with vs. without the Aug 17-27 bars. **Decisive**:
  removing that one episode (2.4-9.6% of each sample) flips BTC and SOL negative at
  *every single* interval/lookback with no exceptions, and roughly halves or flips ETH's
  numbers too (best surviving ETH cells: 1h lb=2 Sharpe 0.85, 2h Sharpe 0.80-0.96 — mild,
  inconsistent, not a confirmed edge). The episode alone has Sharpe 5-16 in isolation.
  **Verdict revised from "unresolved" to killed**, same confidence level as 15m now: the
  entire apparent 30m-2h edge, for all three coins, was this one rally episode.

**H_OFI — order-flow imbalance (taker buy vs. sell volume) predicting the immediately
following price move.** First hypothesis in this project's Phase 1 to use the
collector's own tick-level trade data (`data/live/date=*/kind=trade/`) rather than
candles or an external feed — real Hyperliquid aggressor-side trades, ~4.5M for
BTC/ETH/SOL over the ~3.6 days the collector happened to be running (2026-08-30 to
09-03). **Found the collector stopped writing on 09-03 and was never restarted** —
flagged here rather than silently worked around; this tests the real data that exists,
not a live feed, and more history would need a restart
(`nohup .venv/bin/python -m src.collector.main >> logs/collector.log 2>&1 &`).
Non-overlapping bins from 15s to 15min (`research/ofi_hypothesis.py`), bin i's signed
volume fraction predicting bin i+1's return, single-asset cost (session 12 real L2
numbers, one leg not two).

**Result: one real, statistically robust microstructure effect found — BTC order flow
imbalance at 15s genuinely predicts the next 15s bin's return (t=4.87, the only cell of
18 to survive Bonferroni, correlation sign positive = continuation/impact, not
reversal).** But it's too small to trade: the raw edge is +0.0006% per bin, against a
session-12 real round-trip cost of 0.00082% even at BTC's most favorable estimate — net
negative. Sign accuracy is only 45.4% (worse than a coin flip) despite the positive
mean-return edge, i.e. this is a magnitude-weighted effect (larger moves land more often
on the correlated side), not a simple win-rate edge — consistent with known
price-impact microstructure (a taker sweep both signals and causes short-term
continuation), not a new discovery, and not exploitable at this account's execution
speed regardless. SOL shows a directionally consistent (all-positive) pattern from 60s to
900s that doesn't individually clear Bonferroni's 18-test bar — worth a second look once
the collector has more days of history, not confirmed now. **Collector restarted same
session** (had been dead since 09-03, silently, for 3+ days) specifically so this
question — genuinely still open, unlike H_TSMOM_GAP's 30m-2h band which the follow-up
below kills outright — has a real chance of being resolved with more than 3.6 days of
tick data next time it's revisited.

**H_FUNDCYCLE — funding-extreme contrarian reversal at Hyperliquid's actual settlement
cadence.** H_B (above) tested this using funding summed to a DAILY signal against daily
forward returns; this asks whether that summing washed out a pattern that exists at the
native cadence, which the user correctly flagged is hourly on Hyperliquid. **Checked
directly rather than assumed**: Hyperliquid's own `fundingHistory` shows an 8h cadence
through 2025-07-19, genuinely hourly only after that (confirmed on BTC's raw funding
timestamps) — restricted this test to the post-transition ~408-day hourly-native regime
only, not a blend of two different settlement mechanisms. Same trailing z-score
contrarian construction as H_B, at 3 lookbacks (24h/72h/168h) × 4 horizons (1h/4h/12h/24h)
× 3 coins = 36 tests (`research/funding_cycle_hypothesis.py`).

**Result: fully null, 0/36 survive Bonferroni, best |t|=1.80** (ETH, 24h lookback, 1h
horizon — nowhere close). Confirms H_B's original daily-resolution null wasn't an
artifact of aggregating over 24 settlements per test point — funding-extreme reversal
just isn't there on this venue for these three coins, at any resolution checked so far.

**Reading all three together**: no new tradable edge found this session, but each result
is informative rather than a dead end — H_TSMOM_GAP's train/test sign-flips plus the
shared-episode catch is a real methodological win (this project now knows to check any
future 2026-07-through-09 test window against the same confound before trusting it);
H_OFI confirms real microstructure exists on this venue but isn't accessible at this
account's speed/size; H_FUNDCYCLE closes out funding-level reversal as a mechanism
regardless of resolution. H_TSMOM (locked ETH 8h combo) remains the only validated edge
in this project.


---

### H_JOINT_PORTFOLIO — the 4 locked combos run pooled on one account, not independently (session 21, user-prompted: "now we are running multiple combos at the same time")

Every `p_pass`/$-per-month number computed so far (deployment-config, funded-phase sim)
simulates ONE combo alone, using the full account, with its own private barrier pair.
That's not what's live: `tsmom_beta_live.py` splits `availableBalance` evenly across all
4 locked combos (BTC 4h, ETH 8h, ETH 12h, SOL 12h) simultaneously — one account, one
pooled equity curve, one pair of barriers, fed by 4 quarter-sized, imperfectly-correlated
return streams. Nothing had simulated that pooled structure before. New file:
`research/tsmom_joint_portfolio_barrier_sim.py`.

**Method:** each combo's own already-validated per-bar strategy-return series
(`tsmom_walkforward._strat_ret_series`, restricted to that combo's own walk-forward OOS
test slice), scaled by its locked leverage k and by 0.25 (matching the live allocation),
resampled to daily. The 4 daily series inner-joined on calendar date → common overlap
2025-12-25 to 2026-08-31 (250 days, bounded by BTC 4h's own already-short OOS window, not
newly introduced by pooling). Joint block-bootstrap draws the SAME random day-blocks for
all 4 series at once — preserves real historical same-day co-movement (pairwise daily
correlation 0.50-0.75 full-sample, 0.38-0.70 with the Aug rally excised) — instead of the
independence every single-combo sim implicitly assumed.

**Headline, then the outlier check the project's own convention (H_ETF/H_LEADLAG/
H_TSMOM_GAP) requires before trusting a big positive surprise:**

| | Pooled ann. Sharpe | p_pass (barrier sim) | median days to resolution | $/mo @ $100K (renewal sim) |
|---|---|---|---|---|
| Full 250d sample | 2.07 | **81.0%** | 83 | $1,630 |
| Ex Aug 17-27 rally (10/250 days) | 1.11 | **63.6%** | 120.5 | $838 |

The already-flagged Aug 17-27 rally episode (H_ETF's original fragile episode, also what
killed H_TSMOM_GAP's 30m-2h band) sits inside this sim's 250-day window too and inflates
the headline meaningfully — same discipline, same result: don't trust the number before
excision. **But unlike H_TSMOM_GAP's 30m-2h band, this one survives excision** — 63.6%
pooled p_pass and $838/mo are still real, and both are well above any single locked
combo's own number (best individual combo: ETH 8h, 36.0% p_pass, ~$780/mo scaled to
$100K). The diversification benefit is genuine, not just outlier-driven, because pairwise
correlation (0.38-0.70 ex-episode) is well below 1 — pooling 4 signals that don't move in
lockstep smooths the combined equity path enough to meaningfully raise the odds of
reaching +9% before -3%, even after removing the one episode that made it look dramatic.

**Caveats, stated plainly, not yet resolved:**
- 250 days of common history (repeated many times per simulated 3-year path via block
  bootstrap) is a short source for a correlation estimate this load-bearing — same
  general caution as every block-bootstrap p_pass in this project, sharper here because
  the number the user actually cares about (the *current* live account's odds) hangs on
  it directly.
- Daily-resolution barrier checking loses intraday breach timing (same caveat H_STRUCT
  flagged for its own daily baseline) — likely a small further inflation of p_pass.
- Quarter-weighting (0.25/combo) is `tsmom_beta_live.py`'s choice for paper-trader
  comparability, not an optimized allocation — this answers "what does the *current*
  split deliver," not "what split is best." Not explored: whether an unequal weighting
  (e.g. more to ETH 8h, the strongest solo combo) beats the naive equal quarter-split.
- This is a backtest/bootstrap answer, not a live-tracking one — the live Beta account
  (queried directly this session) is 3 days into its first cycle, nowhere near enough
  history to compare against either the 81% or 63.6% figure yet.

**Unequal weighting toward the strongest solo combo (ETH 8h), tested out of curiosity —
hurts, doesn't help.** 4-point grid, `weights` param added to `build_daily_series`:

| Weighting (BTC4h/ETH8h/ETH12h/SOL12h) | Sharpe | p_pass full | p_pass ex-episode | $/mo @ $100K ex-episode |
|---|---|---|---|---|
| Equal 25/25/25/25 | 2.07 | 81.0% | **63.6%** | $838 |
| Tilt 15/40/25/20 | 1.97 | 76.8% | 58.0% | $860 |
| Heavy 10/55/20/15 | 1.85 | 72.8% | 52.4% | $861 |
| ETH 8h only 0/100/0/0 | 1.51 | 60.6% | 39.4% | $895 |

p_pass falls monotonically as ETH 8h's weight rises, converging on its own solo number
(39.4% ex-episode ≈ the ~36% from the original per-combo table) as weight → 100%. $/mo
barely moves (concentrating raises expected return per dollar but not enough to offset
the diversification loss). Coarse grid (4 points, not swept), but the direction is
monotonic and consistent enough to trust qualitatively: **equal-weighting is close to
right, not a naive placeholder** — the barrier-hitting metric rewards path smoothness
(diversification) more than raw expected return, and the 4 signals' correlation (0.38-
0.70) is low enough that spreading equally captures most of the available benefit.
Untested: weightings that *underweight* the noisier legs symmetrically, or a proper
grid/optimizer search — not run given the effort already spent and the clear qualitative
answer to the actual question asked (does tilting toward the best solo combo help — no).


---

### H_BUDGET_RUIN and H_SHARPE_SE — business-decision math for the real $5K Turbo
purchase (session 22, user-prompted): savings-ruin/timing simulation, the standard error
of a rolling Sharpe, two concurrent accounts on the same signal, and a live-execution
gap the discussion surfaced

Three linked questions from the user, all with the same underlying goal: replace gut
feeling with an actual number before committing real money. **The user's own framing,
confirmed correct by all three analyses below: each real challenge purchase is an
approximately independent draw (no probability "carries over" from a prior pass/fail);
what changes with more paid or free attempts is the sample size behind the strategy's
track record, not the odds of any single draw.**

**1. Personal-savings ruin/timing simulation (`research/budget_ruin_sim.py`).** Reuses
`tsmom_joint_portfolio_barrier_sim.py`'s pooled-path bootstrap unmodified (same 4 locked
combos, same quarter-weighting as `tsmom_beta_live.py`) at $5K/$25-fee scale instead of
$100K/$450, with a hard personal budget ceiling (~$637, the user's ~€590) and a
"fail → pay again from savings, pass → stop" renewal loop, 4000 sims:

| | Full 250d sample | Ex Aug 17-27 rally |
|---|---|---|
| P(exhaust the whole budget before ever passing) | 0.0% | 0.1% |
| Fails before the eventual pass (median / p95) | 0 / 1 | 0 / 2 |
| $ spent on fees before passing (median / p95) | $25 / $50 | $25 / $75 |
| Calendar days to first pass (median / p5-p95) | 105 / 15-337 | 197 / 48-639 |

At a $25 fee against a ~$637 budget (25 affordable back-to-back fails), ruin is a
non-issue even at the more conservative per-attempt pass rate (~64%) — most simulated
savers pass on their very first purchased attempt. **The real constraint is calendar
time, not money**: median wait to a first pass is 3.5-6.5 months, not the 2-3 months
initially assumed, with a real (if unlikely) tail past a year. Caveat carried over
unchanged from H_JOINT_PORTFOLIO: this trusts the historically-bootstrapped p_pass
distribution holding forward — it cannot see a genuine future regime break, only
quantify risk under "the backtested distribution keeps behaving like it has."

**2. How much sample a "solid" rolling Sharpe needs (`research/sharpe_se_analysis.py`).**
Standard result (Lo, 2002): for daily-resampled returns, `SE(annualized Sharpe) ≈
sqrt((1 + SR²/730)/T)` ≈ `1/√T` at these Sharpe levels, T = years of history. Computed
against this project's own measured Sharpes (1.11 pooled ex-episode, 1.24 best solo
combo, 0.41 weakest tested combo):

| Horizon | SE (SR=1.11) | 95% CI | Excludes 0? |
|---|---|---|---|
| 1 week (current Beta run) | 7.23 | [-13.1, 15.3] | no |
| 90 days (this project's own proposed check window) | 2.02 | [-2.8, 5.1] | no |
| 6 months | 1.42 | [-1.7, 3.9] | no |
| 1 year | 1.00 | [-0.9, 3.1] | no |
| 2 years | 0.71 | [-0.3, 2.5] | no |
| 4 years | 0.50 | [0.1, 2.1] | yes (barely) |
| 9 years | 0.33 | [0.5, 1.8] | yes |

**Honest conclusion: no realistic monitoring window — weeks, months, even a couple of
years — makes a rolling Sharpe statistically "solid."** This is a lower bound too (i.i.d.
assumption; TSMOM's ~10-bar average hold means real serial correlation makes the true SE
somewhat worse). The rolling-Sharpe check already on the Phase 4 punch list should be
understood as a **smoke detector for gross breakdowns** (a long, unambiguous negative
stretch), never as a **statistical test** capable of confirming the edge intact or
detecting moderate decay — asking it to do the latter within any patience-compatible
timeframe asks more of the data than exists.

**3. Two accounts running the identical strategy concurrently: no capacity/edge decay,
but effectively zero pass/fail diversification.** No market-impact concern at $5K-$100K
notional against Hyperliquid's depth, so the per-dollar edge doesn't degrade. But this is
the mirror image of H_JOINT_PORTFOLIO's real diversification finding: that result works
*because* the 4 combos are only partially correlated (0.38-0.70 daily). Two accounts
running the *same* 4-combo mix concurrently are correlated at ~1 — it's the same trades
at different notional, not a second independent draw. P(at least one passes) is nowhere
near the naive `1-(1-p)²`; realistically both accounts pass or both fail together. Actual
diversification across two accounts would require staggering start times (drawing on a
different slice of realized market conditions), not running both from day one.

**4. Live-execution status, checked directly against the logs (not assumed).** A real
bug was already caught and fixed by Beta testing exactly as intended: 13 consecutive
`ORDER FAILED [400] order_side_must_align_with_position_side...` errors early in
`logs/tsmom_beta_live.log`, root-caused in the code itself — Propr holds ONE aggregate
position per coin, not per strategy, so ETH 8h and ETH 12h (sharing the same underlying
ETH position) were being synced independently and colliding; fixed by netting all
combos per coin before syncing (see the comment above `main()` in
`tsmom_beta_live.py`). Confirmed exercised since: bar-close detection, signal computation
across all 4 combos, idempotent leverage-setting, dry-run mode, active-challenge
discovery, several UTC daily-reset boundaries with no incident. **Not yet exercised**:
the ATR trailing stop actually firing, a barrier (+9%/-3%) actually resolving on this
pooled config, opposite-direction netting (two combos on the same coin disagreeing, as
opposed to the same-direction case that produced the bug above), a single combo's own
long↔short flip, and Propr-side API errors on the order/account path itself (the one
transient error seen, a 502, was Hyperliquid's public data API in the paper trader, not
Propr's order path). **One concrete gap surfaced by this discussion, directly relevant to
running two accounts (point 3 above)**: `tsmom_beta_live.py`'s `main()` does
`client.account_id = active[0]["accountId"]` — it only ever trades the *first* active
challenge attempt the API returns. If two attempts are simultaneously active (the real
$5K Turbo alongside the continuing Beta account, or later $5K + $100K), the script has no
logic to run both — it would silently trade only whichever one the API lists first. Open,
not yet fixed.


---

### H_GIVEBACK — trade-level give-back / tail-concentration diagnostic (session 23,
user-prompted from watching the live Beta charts: the strategy appears to give back a
large slice of unrealized gain in most trades before its ATR stop fires, with total PnL
looking carried by a handful of monster-runner trades)

H_TSMOM's existing outlier check (session 10, above) is BAR-level — drop the 20
largest-magnitude bars, see if aggregate Sharpe survives — and found TSMOM does *not*
have H_ETF's tail-concentration problem at that granularity. That is a different question
from this one: does each individual TRADE give back most of its own peak unrealized gain
before it closes, and is total realized PnL concentrated in a small number of trades?
Nothing in the repo had measured this before. New file:
`research/tsmom_giveback_diagnostic.py`.

**Method:** duplicate `_strat_ret_series`'/`backtest_trailing_stop`'s exact entry/ATR-stop
loop bar-for-bar (same duplicate-rather-than-touch-validated-code convention as every
other script here), emitting one row per closed trade (entry/exit price & time, direction,
size, realized log-return) plus MFE — max favorable excursion, tracked causally using the
same intrabar high/low convention the ATR stop itself checks breaches against. MFE is a
retrospective property of an already-closed trade, computed after the fact for diagnostic
purposes only — never fed back into a live decision, so this is not the look-ahead failure
mode session 10's bug was. `giveback_ratio = (mfe_ret - realized_ret) / mfe_ret`, defined
only where `mfe_ret > 0` (trades never favorable get their own bucket, not a fabricated
0/1). **Reconciliation check, required before trusting anything below**: replaying
`extract_trades`' own per-bar return array through the exact same `(1+x).cumprod()` equity
convention `backtest_trailing_stop` uses reproduces its reported `total_return` bar-for-bar
on all 4 locked combos (89.4%/184.7%/235.7%/223.5% — exact match) — confirms the new
extraction loop is faithful to the validated one, not a subtly different process.

**Headline: the give-back pattern is real, and it is broad, not a couple-of-trades
artifact.**

| | n trades | % never favorable | median give-back | % give-back > 50% | % give-back > 80% |
|---|---|---|---|---|---|
| BTC 4h | 246 | 0.0% | 136.4% | 87.8% | 68.7% |
| ETH 8h | 200 | 0.5% | 129.3% | 88.4% | 68.8% |
| ETH 12h | 134 | 0.0% | 103.5% | 85.8% | 63.4% |
| SOL 12h | 135 | 0.0% | 104.6% | 82.2% | 63.7% |
| **Pooled** | 715 | 0.1% | **119.0%** | 86.6% | 66.8% |

Median give-back ratio *exceeds 100%* pooled — meaning the median trade that ever reaches
positive unrealized PnL still closes as a net **loser** by the time its ATR stop fires
(consistent with the strategy's ~43% overall win rate: 308/715 trades pooled close
profitable). This is the classic trend-following payoff shape stated precisely for the
first time in this project: most trades are false starts that reverse into a small-to-
moderate loss after a brief favorable excursion, and the net-positive total return is
carried by the minority that don't reverse.

**PnL concentration (winning trades only):**

| | n winners | top 5% share | top 10% share | top 20% share | Gini |
|---|---|---|---|---|---|
| BTC 4h | 94 | 37.2% | 50.5% | 64.9% | 0.61 |
| ETH 8h | 84 | 31.3% | 45.4% | 62.1% | 0.58 |
| ETH 12h | 65 | 28.0% | 42.1% | 59.5% | 0.56 |
| SOL 12h | 65 | 27.3% | 39.2% | 55.4% | 0.53 |
| **Pooled** | 308 | 28.8% | 43.9% | 61.6% | 0.58 |

**Outlier-robustness check (the project's own convention, applied here at trade level):**
dropping the top 5 or top 10 largest-magnitude trades pooled barely moves the picture —
median give-back 119.0% → 121.4% → 121.8%, top-10%-share 43.9% → 40.9% → 37.4%, Gini 0.58 →
0.56 → 0.53. **This is the key distinction from H_ETF/H_TSMOM_GAP-style fragility**: total
PnL here is meaningfully concentrated in the better-performing trades (Gini ~0.53-0.61,
top 10% of winners ≈ 40-50% of winning PnL) but is not *dominated* by a handful of them —
removing the 10 largest trades leaves the concentration and give-back numbers materially
unchanged, the opposite of H_ETF's two-episode collapse. **Train/test temporal stability**
(walk-forward's own bar_range split, no parameter tuned here): give-back median is close
between train and test for all 4 combos (BTC 4h 141.3%→126.3%, ETH 8h 129.3%→141.3%, ETH
12h 110.9%→92.1%, SOL 12h 104.7%→99.7%) — the pattern isn't unique to one sub-period.

**Verdict: give-back is real, moderate-to-severe, and structurally broad rather than
concentrated in a few trades** — option (b) from the pre-registered verdict categories.
This supports a take-profit-based mechanism being *viable in principle* (there's a broad,
recurring pattern to capture, not a couple of outliers a TP would just cap away), while
the meaningful-but-not-dominant PnL concentration (Gini ~0.58, not ~0.9+) is the caution
flag for session 24/25's capped-TP test: capping gains too aggressively risks trading away
real tail contribution, not just noise. Caveat: `giveback_ratio` and MFE are computed in
log-return space (summed per-bar components, which telescope exactly to
`log(exit_px/entry_px)` for constant position size) — a different, more standard
convention than `backtest_trailing_stop`'s own `total_return`/equity metric, which
compounds those same log-return-scaled quantities arithmetically via `(1+x).cumprod()`
(an existing quirk of the validated code, left untouched); the two are reconciled above
via the bar-series replay, not by comparing the two return conventions directly. Full
per-trade data written to `research/output/tsmom_giveback_trades.json` for reuse by
session 24/25's TP-threshold and ratchet-schedule design.


---

### H_DIRECTIONAL — finishing the session-22 cancellation study (session 23): does
BTC/ETH/SOL long-short cancellation actually cost pooled pass-probability?

`research/tsmom_directional_variants.py` was written session 22 (user-prompted from a
live Beta observation: "BTC short / ETH+SOL long often cancel each other and the pooled
strategy remains flat") but never run and written up. No new code this session — just
running it and holding its output to the same rigor bar (ex-episode excision) as
H_JOINT_PORTFOLIO, via a throwaway reporting-time slice (not a change to the validated
script) that reuses its `build_daily_series_variant`/`VARIANTS` unmodified.

**Reconciliation check**: the `long_short` variant (current production signal) reproduces
H_JOINT_PORTFOLIO's already-published full-sample numbers essentially exactly — Sharpe
2.07, p_pass 81.0% — confirming the two scripts haven't drifted.

**Netting diagnostic (full history, both-directions signal, 833/834 active days):** mean
fraction of gross exposure surviving netting = **69.3%** (so ~31% of gross exposure
cancels on an average active day) — a real, common effect, matching what the user watched
happen live. But **complete cancellation is rare**: fully-flat days (net exposure ≈ 0
while gross > 0) are only 0.4% of active days; on 52.2% of active days all 4 combos
actually agree in direction and nothing cancels at all. The user's live-observed "the
strategy remains flat" moments are real but atypical, not the modal case.

**Three-variant comparison, full sample and ex Aug 17-27 rally (matching
H_JOINT_PORTFOLIO's own excision window; ex-episode n=239 days here vs. 240 there, a
one-day boundary difference, close enough not to affect the conclusion):**

| variant | corr range (full) | corr range (ex-rally) | Sharpe (full) | p_pass (full) | Sharpe (ex-rally) | p_pass (ex-rally) |
|---|---|---|---|---|---|---|
| long_short (production) | 0.50-0.75 | — | 2.07 | **81.0%** | 1.09 | **62.2%** |
| long_only | 0.70-0.91 | — | -0.15 | 23.9% | -1.66 | 0.3% |
| short_only | 0.78-0.95 | — | 1.05 | 46.2% | 1.42 | 55.5% |

**Verdict: restricting direction does not help pooled pass-probability — it hurts, and
the mechanism is exactly the one the project's diversification finding (H_JOINT_PORTFOLIO's
unequal-weighting sweep) already predicted.** Cross-combo correlation *rises* under
directional restriction (0.70-0.95 for long_only/short_only vs. 0.50-0.75 for long_short)
— forcing all 4 combos to agree on direction makes them move together more, not less,
during broad market moves, which is the opposite of what the "cancellation is costing us
return" intuition would suggest. `long_only` is uniformly the worst variant and gets
*catastrophically* worse ex-rally (Sharpe -1.66, p_pass 0.3%) — the Aug rally was propping
up an otherwise deeply negative long-only result, not representative of its typical
behavior. `short_only` is closer to competitive (and actually improves ex-rally, since
removing an up-rally naturally helps a short-only book) but still meaningfully behind
`long_short` on both metrics in both windows. **The BTC/ETH/SOL cancellation the user
observed live is real (69.3% average netting survival) but is a cost of genuine
diversification benefit, not a fixable inefficiency** — the same conclusion
H_JOINT_PORTFOLIO already reached from the weighting side, now confirmed from the
directional-restriction side too. No further action recommended on this thread; the
production long_short signal and current 4-combo universe should not change.


---

### H_TP_EXIT / H_WIDE_TRAIL — two re-derived exit-mechanism variants (session 24)

A recollection that a capped take-profit was tried and killed, and a wider "let runners
run" trailing stop was found to be a false positive on more rigorous testing, did not
survive a full-text search of this repo (session 23 preamble, H_GIVEBACK) — the current
ATR-2.5x stop was in fact validated (session 11) as this project's most walk-forward-
robust result. This session builds and tests both ideas honestly from scratch, informed
by H_GIVEBACK's empirical MFE distribution (raw MFE magnitude varies enormously by combo
in percent terms — BTC 4h median ~2.8%, SOL 12h median ~13.1% — so both variants are
parameterized in ATR-multiples, not raw percent, for unit consistency with the existing
stop). New file: `research/tsmom_exit_variants.py`, same causal-timing discipline as
every stop/TP check in this project (level fixed using data through t-1, breach checked
against bar t's own intrabar high/low).

**Sanity checks (both must pass before trusting anything below, both did):**
`vol_target_returns_ratchet` with schedule `[(0.0, 2.5)]` reproduces `vol_target_returns`
(atr_mult=2.5) bar-for-bar on all 4 combos; `vol_target_returns_with_tp` with the TP
effectively disabled (`tp_atr_mult=1e6`, mode="alt") does the same.

**H_TP_EXIT — take-profit fixed at entry, ATR-multiple grid {1.0, 1.5, 2.0, 2.5, 3.5},
selected by train-only Sharpe, walk-forward tested:**

| coin/interval | mode | selected tp (×ATR) | train Sharpe | test Sharpe | test p_pass | outlier-dropped Sharpe | baseline test Sharpe |
|---|---|---|---|---|---|---|---|
| BTC 4h | alt | 3.5 | 0.43 | 0.73 | 23.0% | **-0.02** | 1.95 |
| BTC 4h | tp_only | 3.5 | 0.93 | **-0.36** | 24.8% | 0.55 | 1.95 |
| ETH 8h | alt | 2.0 | 0.55 | 0.53 | 29.2% | 0.41 | 1.54 |
| ETH 8h | tp_only | 3.5 | 0.84 | **-0.08** | 24.5% | 0.49 | 1.54 |
| ETH 12h | alt | 3.5 | 1.46 | 1.28 | 29.8% | 1.00 | 1.54 |
| ETH 12h | tp_only | 1.0 | -0.26 | 0.33 | 23.2% | -0.10 | 1.54 |
| SOL 12h | alt | 2.5 | 1.57 | **1.53** | 26.2% | 1.30 | 1.32 |
| SOL 12h | tp_only | 3.5 | 0.68 | **-0.04** | 26.2% | -0.06 | 1.32 |

**Verdict: `tp_only` (TP replacing the ATR stop entirely) is a clean kill** — negative or
near-zero test Sharpe on every combo, uniformly worse than the locked baseline. `alt`
mode (TP alongside the existing stop, whichever triggers first) is mixed and, at best,
marginal: roughly flat-to-worse for BTC 4h/ETH 8h/ETH 12h, and the one apparent win (SOL
12h, 1.53 vs. baseline's 1.32) is small and **BTC 4h's own `alt` result collapses under
the bar-level outlier check** (0.51 in-sample → **-0.02** after dropping the 20 largest
bars) — exactly the fragility pattern this project's convention exists to catch. No TP
variant here beats the locked baseline convincingly and robustly on any combo.

**H_WIDE_TRAIL — fixed-wider multipliers {3.5, 4.0, 5.0} and two 2-level ratchet
schedules, selected by train-only Sharpe, walk-forward tested:**

| coin/interval | selected | train Sharpe | test Sharpe | test p_pass | outlier-dropped Sharpe | baseline test Sharpe |
|---|---|---|---|---|---|---|
| BTC 4h | **baseline itself (2.5x)** | 0.89 | 1.95 | 24.2% | 0.26 | 1.95 |
| ETH 8h | ratchet 2.5→5.0 @3×ATR | 1.03 | 1.52 | 25.0% | 1.12 | 1.54 |
| ETH 12h | **baseline itself (2.5x)** | 1.09 | 1.54 | 31.2% | 0.69 | 1.54 |
| SOL 12h | ratchet 2.5→4.0 @2×ATR | 1.26 | **0.66** | 23.8% | 0.89 | 1.32 |

**Verdict: no combo shows a robust improvement.** For BTC 4h and ETH 12h, train-only
selection picks the *current production stop itself* out of the candidate grid — none of
the wider/ratchet variants beat it even in-sample. ETH 8h's selected ratchet is
essentially indistinguishable from baseline (1.52 vs. 1.54 test Sharpe). **SOL 12h
reproduces, for the first time in this repo, the exact "looks good in training, fails
out-of-sample" pattern the user recalled**: train Sharpe 1.26 (better than baseline's
own train performance) but test Sharpe collapses to 0.66, well below baseline's 1.32 —
a genuine selection-artifact/overfitting story, even though no such result previously
existed in the repo to have produced that recollection. Read charitably, this session's
finding is a plausible origin for the "false positive" memory, just not one this repo had
actually run before now.

**Combined verdict for both variants, feeding into H_POST_TP_REENTRY (session 25/26):**
neither a standalone capped TP nor a wider/ratcheting stop robustly beats the locked
ATR-2.5x baseline as a *terminal* exit rule — the current production stop remains the
right default. The `alt`-mode TP thresholds selected here per combo (BTC 4h 3.5×ATR, ETH
8h 2.0×ATR, ETH 12h 3.5×ATR, SOL 12h 2.5×ATR) are still reused in H_POST_TP_REENTRY purely
as **stage-1 triggers** for a two-stage mechanism, not as validated terminal exits in
their own right — a deliberately distinct claim from what this section tested.


---

### H_POST_TP_REENTRY — the post-TP second-decision mechanism (session 25) — the actual
novel research question this thread was built for

**Motivation**: H_GIVEBACK (session 23) confirmed the user's live observation is real and
broad — the median trade that ever reaches favorable unrealized PnL still closes as a net
loser (119% pooled give-back ratio). The idea: keep the entry signal unchanged, add a
take-profit as a first-stage exit, and once it fires, take a genuinely independent second
decision — re-enter same direction (momentum holds), flip (reversal), or go flat until the
next primary bar-close signal — instead of the give-back happening unchecked, or a plain
TP just going flat and missing any continuation.

**Design fix, resolved via a clarifying question before writing any code**: the window
from original entry to TP-fire is favorable *by construction* — a score defined as
"return over that whole window" can never be negative, making "flip to reversal"
structurally unreachable. Resolved by computing the second-stage score as a **fresh
short-lookback momentum reading** (same style as the base signal, freshly evaluated at
the TP-fire bar) rather than the whole entry-to-TP window's own return — this can
genuinely point either direction. New file: `research/tsmom_post_tp_reentry.py`.
Mechanics deliberately **not chained**: a re-entry/flip trade watches only its own ATR
stop (no TP), so at most one extra decision follows any given TP-fire, not an open-ended
sequence. TP threshold is **not re-searched** — fixed per combo to session 24's
train-selected `alt`-mode value (a stage-1 trigger only, not an endorsement of TP as a
terminal exit). What *is* searched, train-only per combo: score type (raw log-return vs.
vol-normalized), the short lookback the score is computed over (2/3/5 bars), and the
flat-band threshold `eps`.

**Per-combo walk-forward (selected param vs. locked baseline):**

| coin/interval | selected score | train Sharpe | test Sharpe | test p_pass | baseline test Sharpe | baseline p_pass |
|---|---|---|---|---|---|---|
| BTC 4h | raw, lb=2, eps=0.01 | 1.79 | **0.28** | 22.2% | 1.95 | 24.3% |
| ETH 8h | raw, lb=2, eps=0.01 | 1.43 | **0.34** | 20.2% | 1.54 | 29.3% |
| ETH 12h | raw, lb=2, eps=0.005 | 1.40 | 1.69 | 29.5% | 1.54 | 31.2% |
| SOL 12h | raw, lb=2, eps=0.0 | 0.87 | 1.42 | 26.5% | 1.32 | 24.8% |

BTC 4h and ETH 8h show a clear overfitting/selection-artifact pattern (train Sharpe far
exceeds test — 1.79→0.28, 1.43→0.34), both notably worse than baseline OOS. ETH 12h and
SOL 12h show small, real-looking single-combo improvements (1.69 vs. 1.54; 1.42 vs. 1.32).
Every combo's walk-forward search picked the *shortest* lookback tested (2 bars) — worth
flagging on its own: this project already established (H_TSMOM_FAST, session 18) that
TSMOM-style momentum evaluated at very short horizons is "uniformly, catastrophically
negative" due to reversal-dominance. A 2-bar fresh-momentum score is operating in exactly
that already-proven-dead territory, which is the most likely root cause of the mixed/poor
result: the second-stage decision is largely trading on noise at a horizon where this
signal family has no edge, not a cheap way to capture the intra-window move.

**Portfolio-level benchmark — the single most decision-relevant number for the user's
actual question, all 4 locked combos pooled at the current 25%-each weight, same
machinery as H_JOINT_PORTFOLIO/H_DIRECTIONAL:**

| | Pooled ann. Sharpe | p_pass | fail_static |
|---|---|---|---|
| **Locked baseline (H_JOINT_PORTFOLIO)**, full sample | 2.07 | **81.0%** | 19.1% |
| **H_POST_TP_REENTRY**, full sample | 0.77 | **58.5%** | 40.5% |
| **Locked baseline**, ex Aug 17-27 rally | 1.11 | **63.6%** | — |
| **H_POST_TP_REENTRY**, ex Aug 17-27 rally | -0.11 | **20.6%** | 75.2% |

Cross-combo daily-return correlation drops sharply under the new mechanism (0.09-0.39 vs.
baseline's 0.50-0.75) — the extra idiosyncratic re-entry/flip trades genuinely decorrelate
the 4 combos further, which by itself is the *direction* that helps diversification
(H_JOINT_PORTFOLIO's own finding). **But the underlying per-combo return quality is hurt
enough (BTC 4h/ETH 8h's collapse dominates the pool, both getting badly overfit results)
that pooled p_pass falls hard anyway — from 81.0%→58.5% full sample and, more
importantly, from 63.6%→20.6% ex the one big rally episode.** The ex-rally comparison is
the fairer one (the full-sample numbers on both sides are inflated by the same episode),
and it shows an unambiguous, large regression, not a marginal one.

**Verdict: REJECT, this implementation of the mechanism should not be pursued as
tested.** It does not achieve the goal of capturing intra-window moves without hurting
the strategy — it substantially degrades pooled pass-probability, most severely once the
one dominant rally episode is excluded. The most likely fixable root cause is the score's
lookback: every combo's walk-forward selection converged on the shortest (2-bar) option,
landing the second-stage decision in a horizon this project has already shown TSMOM has
no edge at. **Not yet tried, and worth a follow-up if this thread is revisited**: a
score-lookback grid extending well beyond 5 bars (closer to the base signal's own
multi-bar horizon), which the current grid didn't test and which the "shortest always
wins" pattern suggests may simply not have been given a fair chance here. Until that's
tested, the honest conclusion is that this specific mechanism, as built and validated
this session, should not replace or augment the locked production strategy.

**CORRECTION (session 26): the REJECT verdict above was computed with a real sign bug,
now fixed.** `vol_target_returns_post_tp`'s hold/flip branching thresholded the fresh
short-lookback score directly (`chosen > eps -> hold`, `chosen < -eps -> flip`), but
`chosen` is an ABSOLUTE reading (positive iff price rose over the score window,
regardless of `pos_dir`) — this mapping is only correct for a long trade (`pos_dir=+1`)
and is **backwards for a short trade** (`pos_dir=-1`): a short trade's TP fires on a price
*fall*, so momentum genuinely continuing down at that moment (`chosen < 0`) should mean
"hold" (stay short), but the buggy code read that as `chosen < -eps -> forced_dir =
-pos_dir` and flipped to long instead. Since the production signal trades both directions
roughly equally, this corrupted close to half of all second-stage decisions. The
sanity check that ran before the session-25 numbers were trusted used `tp_atr_mult=1e6`
(TP disabled) specifically to verify baseline-reproduction — it never exercised the
buggy branch at all, so it could not have caught this. Fixed by comparing `pos_dir *
chosen` against `±eps` instead of `chosen` directly.

**Corrected results are dramatically different — the mechanism is now roughly ON PAR
with the locked baseline, not a clear rejection:**

| coin/interval | selected score (corrected) | test Sharpe | baseline test Sharpe |
|---|---|---|---|
| BTC 4h | raw, lb=5, eps=0.0 | **2.02** | 1.95 |
| ETH 8h | vol_norm, lb=2, eps=0.25 | 1.55 | 1.54 |
| ETH 12h | vol_norm, lb=2, eps=1.0 | 1.43 | 1.54 |
| SOL 12h | vol_norm, lb=3, eps=0.5 | 1.09 | 1.32 |

| | Pooled ann. Sharpe | p_pass |
|---|---|---|
| Locked baseline, full sample | 2.07 | 81.0% |
| H_POST_TP_REENTRY (corrected), full sample | 1.91 | **82.2%** |
| Locked baseline, ex Aug 17-27 rally | 1.11 | 63.6% |
| H_POST_TP_REENTRY (corrected), ex Aug 17-27 rally | 1.04 | **64.1%** |

Two combos now improve (BTC 4h, ETH 8h), two are modestly worse (ETH 12h, SOL 12h) — but
the portfolio-level number, the one that actually answers the user's question, comes out
essentially tied with baseline on both cuts, marginally *better* on the fairer ex-rally
comparison (64.1% vs. 63.6%). Cross-combo correlation is also more sensible now
(0.34-0.71, vs. the buggy version's suspiciously low 0.09-0.39 — the bug was scrambling
half the trades' direction, which artificially decorrelated the combos without adding
real diversification benefit).

**Revised verdict: NEUTRAL, not REJECT.** The mechanism as corrected doesn't yet clear a
bar of "clearly better," but it's no longer a clear loser either — it's a legitimate
foundation to refine rather than a dead end. This directly motivates session 26's next
step: the fresh short-lookback *price* score converges on very short lookbacks (2-5 bars)
for 3 of 4 combos, the same territory H_TSMOM_FAST already showed has no pure-price-momentum
edge — the natural next question is whether a genuinely different information channel
(volume/participation confirmation, or a trend-persistence regime filter) does better
than re-scoring the same price series at a shorter horizon.


---

### H_POST_TP_SCORE_V2 — volume/participation confirmation and a Hurst-style
trend-persistence gate (session 26), independently researched (web search) rather than
guessed

Search covered: order-flow/trade-flow imbalance (Cont-Kukanov-Stoikov — the textbook
"short-term momentum" measure in microstructure literature, with documented crypto-perp
evidence), funding-rate/OI crowding extremes (literature frames these as more of a
*reversal* signal at crowded extremes than continuation), liquidation-cascade
continuation, volume/participation confirmation (Chaikin-Money-Flow-style), and Hurst-
exponent trend-persistence filters. **Data reality check first**: this project's live
collector (`src/collector/`) has only 9 days of L2-book/trade/funding/OI history as of
this session — nowhere near enough for a walk-forward test of order-flow, funding-
crowding, or liquidation signals to this project's own rigor bar. Those are flagged for
later, not attempted now. Volume/participation and Hurst persistence are both
immediately testable on the full ~2.3-year candle history already on disk — and, not
previously noticed, **every candle file already carries `v` (volume) and `n` (trade
count) fields that `tsmom_hypothesis.load_candles` silently drops and no script in this
project has ever used.**

**Method**: new file `research/tsmom_post_tp_score_v2.py`. (a) Chaikin-Money-Flow-style
score: money-flow-multiplier `((close-low)-(high-close))/(high-low)` weighted by volume,
averaged over a short window, bounded [-1,1] — a genuinely different information channel
from price return, not a faster resampling of the same dead signal. (b) A lightweight
variance-ratio Hurst estimate (`H = 0.5 + 0.5*log(VR)/log(q)`, `VR = Var(q-bar
return)/(q*Var(1-bar return))`) used as a **gate**, not a directional score: forces flat
instead of acting on a hold/flip signal when the recent path isn't confirmed trending
(H≤0.5). CMF needed the market-tuple/block-bootstrap machinery extended with a volume
channel (`extract_market_tuples_v2`/`block_bootstrap_path_v2`, duplicated rather than
modifying the validated originals) so a synthetic Monte Carlo path's volume stays
attached to its own price action; the Hurst filter is pure-price and needed no such
extension. Grid: `{raw, vol_norm}` (lookback 2/3/5, same as session 25/26) × `{cmf}`
(lookback 5/10/20, more standard CMF windows) × `hurst_gate ∈ {False, True}` — 66
candidates per combo, all train-only walk-forward selected. **Sanity check passed on all
4 combos**: TP disabled (`tp_atr_mult=1e6`) reproduces the plain ATR-2.5x baseline
bar-for-bar regardless of `score_type`/`hurst_gate`.

**Per-combo walk-forward (selected param vs. session-25-corrected vs. locked baseline):**

| coin/interval | selected | test Sharpe | outlier-dropped Sharpe | session-25-corrected test Sharpe | baseline test Sharpe |
|---|---|---|---|---|---|
| BTC 4h | cmf, lb=20, eps=0.0 | **2.04** | **0.17** (fragile) | 2.02 | 1.95 |
| ETH 8h | vol_norm, lb=2, eps=0.25 (unchanged) | 1.55 | 0.88 | 1.55 | 1.54 |
| ETH 12h | cmf, lb=20, eps=0.2 | 1.46 | 1.00 | 1.43 | 1.54 |
| SOL 12h | vol_norm, lb=3, eps=0.5, **hurst_gate=True** | **1.46** | **1.43** (robust) | 1.09 | 1.32 |

CMF wins the walk-forward selection for BTC 4h and ETH 12h; the Hurst gate wins for SOL
12h (a real, and — unlike BTC 4h's CMF pick — outlier-robust improvement: 1.52 in-sample
→ 1.43 after dropping the 20 largest bars, barely moves); ETH 8h's selection is unchanged
from session 25 (neither CMF nor the Hurst gate beat the existing vol-normalized score
there). **BTC 4h's CMF-based gain is the least trustworthy of the four**: full-history
Sharpe collapses from 1.22 to 0.17 after dropping the 20 largest bars — the same
fragility pattern flagged for BTC 4h's TP-alt result in session 24 (H_TP_EXIT), a
recurring signal that BTC 4h's OOS numbers in this thread lean on a small number of large
bars more than the other three combos do.

**Portfolio-level benchmark — the decisive number:**

| | Pooled ann. Sharpe | p_pass |
|---|---|---|
| Locked baseline, full sample | 2.07 | 81.0% |
| Session 25 corrected, full sample | 1.91 | 82.2% |
| Session 26 (this section), full sample | 2.04 | 82.0% |
| Locked baseline, ex Aug 17-27 rally | 1.11 | 63.6% |
| Session 25 corrected, ex Aug 17-27 rally | 1.04 | 64.1% |
| **Session 26 (this section), ex Aug 17-27 rally** | **1.17** | **65.6%** |

**Verdict: a genuine, modest improvement — not a dramatic one, and not evenly earned
across combos.** The ex-rally comparison (the fairer cut, per this project's own
convention) shows pooled p_pass climbing from the locked baseline's 63.6% through
session 25's 64.1% to **65.6%** here — a real, if small, gain that survived the full
rigor pipeline (walk-forward, bar-level outlier check, portfolio-level joint
bootstrap). The improvement is concentrated in SOL 12h's robust Hurst-gated result and
BTC 4h's larger but outlier-fragile CMF result, with ETH 8h contributing nothing new and
ETH 12h still trailing its own baseline slightly. **Recommendation: promising enough to
keep as this thread's leading candidate, but not yet strong or uniform enough to propose
touching the locked live Beta config** — the BTC 4h fragility in particular should be
understood as a real caveat, not glossed over, before any such step. Consistent with
every prior step in this thread, `tsmom_beta_live.py` was not touched.


---

### Holdout locked (session 26) — a genuinely untouched confirmation window, before any
further tuning

Real methodological risk this thread had accumulated by session 26, distinct from and on
top of the 66-candidate multiple-testing exposure already flagged in H_POST_TP_SCORE_V2:
three successive mechanism designs (H_POST_TP_REENTRY, its session-26 correction, and
H_POST_TP_SCORE_V2) were all iterated and evaluated against the *same* fixed historical
walk-forward test slice across sessions 25-26 — a form of researcher-driven repeated-look
leakage that a disciplined per-run train/test split does not, by itself, protect against.
User's explicit decision: **lock a holdout now, do nothing else with it, and let more
live/candle history accumulate before touching it.**

**What's locked**: `research/output/tsmom_holdout_cutoff.json` records, per locked combo,
the exact last bar timestamp used by every session in this thread (BTC 4h: 2026-08-31
20:00; ETH 8h: 2026-08-31 16:00; ETH 12h/SOL 12h: 2026-08-31 12:00 — all four candle files
were last fetched around session 23 and have not been refreshed since, so this is also
simply where the on-disk data currently ends). It also freezes the exact per-combo
H_POST_TP_SCORE_V2 parameter selection (`current_candidate`) as *the* design to confirm
later — not something to re-tune once the holdout is evaluated.

**Rule, stated explicitly so a future session doesn't quietly break it**: no candidate
search, walk-forward split, or design decision may include bars after the recorded
cutoff. Once enough new candle history has accumulated (the candle-fetch pipeline needs
re-running first — these files are static snapshots, not live-updating), the frozen
`current_candidate` design gets evaluated against the holdout bars **exactly once**,
result reported honestly regardless of outcome. A second confirmation run against the
same holdout after seeing the first result defeats the entire point and must not happen.

No backtest, significance test, or new research was run this session — this entry is
scope-limited to locking the boundary, per explicit instruction.

### H_BTC_LEADER — BTC's momentum sign drives ETH/SOL's entry direction (session 27,
user-proposed, during the session-24/H_JOINT_PORTFOLIO-driven research freeze that
otherwise runs until ~2026-11-07 — user judged this a distinct new idea, not a
re-litigation of the live signal, and asked for it explicitly)

**Motivation**: same live-observed BTC/ETH/SOL co-movement that produced H_DIRECTIONAL
(session 22/23), but a different proposed fix. H_DIRECTIONAL restricted each combo's OWN
signal to trade only one direction and found that hurts pooled P(pass) — cross-combo
correlation rises (0.70-0.95 vs. 0.50-0.75) while diversification-driven pass
probability collapses outside trending regimes (81.0% → 23.9%/46.2%). This idea instead
leans into the co-movement directly: BTC keeps its own independent signal unchanged;
ETH's and SOL's entry *direction* is replaced with BTC's causally-aligned momentum sign
(each still runs its own ATR-2.5x trailing stop, vol-targeted sizing, and locked
leverage `k` unchanged). Empirically distinct mechanism, same underlying risk
(forcing agreement across combos can raise correlation and hurt diversification-driven
P(pass)) — measured here, not assumed.

New file: `research/tsmom_btc_leader_variant.py`. BTC's own per-bar direction is
causally aligned onto ETH's/SOL's own (slower) bars via `pandas.merge_asof(direction=
"backward")` — each follower bar gets the most recent BTC direction known at or before
its own open, never a future value. Fed into the same joint-portfolio pooled-barrier
machinery as H_JOINT_PORTFOLIO/H_DIRECTIONAL (same `LOCKED` lb/k table, same
walk-forward OOS test-slice discipline, same quarter-weighting, same
correlation-preserving joint block bootstrap, same seed convention).

**Sanity checks (all passed before trusting anything below)**: the BTC-signal parallel
copy reproduces `tsmom_walkforward._strat_ret_series`'s BTC 4h output bit-for-bit
(5000/5000 bars); zero bars in any of the 4 combos' own OOS test slices fall before
BTC's leader history starts (2024-05-20), so no follower bar in this comparison is ever
flat-filled for lack of leader data; a manual spot-check of ETH_12h's aligned rows
across the Aug 17-27 rally start confirms the matched BTC timestamp is always at or
before the follower's own bar open. The baseline arm reproduces H_JOINT_PORTFOLIO's and
H_DIRECTIONAL's already-published numbers essentially exactly (Sharpe 2.07, p_pass
81.0% full-sample; 62.2% ex-rally; 69.3% netting survival), confirming no drift in the
shared machinery.

**Note on the followers-do-see-BTC-flat diagnostic**: followers see BTC's leader signal
as flat 4.5-6.8% of the time within their own OOS slice — verified to be BTC's own
genuine flat periods (BTC itself sits flat, pos_dir==0, ~4.8-5.3% of the time between a
stop-exit and its own next entry), not a coverage gap or an alignment bug.

**Netting diagnostic (full history)**: mean fraction of gross exposure surviving
netting jumps from **69.3%** (baseline, today's independent signals) to **96.1%**
(btc_leader) — days where all 4 combos agree in direction rise from 52.2% to 94.3% of
active days. This is the mechanism working exactly as intended (ETH/SOL now follow
BTC's direction almost always) — and exactly the effect H_DIRECTIONAL already showed
carries a P(pass) cost.

**Comparison, full sample and ex Aug 17-27 rally (same excision window as
H_JOINT_PORTFOLIO/H_DIRECTIONAL):**

| variant | p_pass (full) | Sharpe (full) | p_pass (ex-rally) | Sharpe (ex-rally) |
|---|---|---|---|---|
| baseline (today, independent signals) | **81.0%** | 2.07 | **62.2%** | 1.09 |
| btc_leader (BTC drives ETH/SOL direction) | 72.1% | 1.69 | 50.5% | 0.78 |

Trade counts (direction changes, full history) shift as expected — ETH_8h drops
401→328, while ETH_12h and SOL_12h both rise toward BTC_4h's own more frequent
flip rate (269→325, 271→325) — consistent with followers now inheriting BTC's own
(faster, 4h-lookback=1) signal cadence instead of their own slower one.

**Verdict: this idea hurts pooled pass-probability, for the same reason
H_DIRECTIONAL already identified from a different angle.** Forcing ETH/SOL to follow
BTC's direction raises realized cross-combo agreement (netting survival 69.3% → 96.1%)
and, net of the resulting loss in diversification, pooled P(pass) falls 8.9 points
full-sample (81.0% → 72.1%) and 11.7 points ex-rally (62.2% → 50.5%), with pooled
Sharpe falling in both windows too (2.07→1.69, 1.09→0.78). The three combos' largely
independent signals already capture most of the achievable diversification benefit;
making ETH and SOL structurally dependent on BTC's signal removes benefit rather than
adding it. **No change recommended to the production signal** — each combo should keep
computing its own independent direction, as it does today. This is a second, mechanism-
distinct confirmation of H_DIRECTIONAL's finding: the BTC/ETH/SOL co-movement the user
watched live is real, but it is the cost of genuine diversification, not something
either restricting or coupling the signals can fix without giving up P(pass) in
exchange.

### H_PARTIAL_SCALEOUT — bank part of a trade's gain without cutting the runner
(session 28, user reframe: this thread stops optimizing P(pass) and instead targets a
more normalized return distribution and faster strategy resolution)

**Motivation, explicit reframe**: this project is not the user's main income and every
dollar risked is treated as tuition; the objective shifts from P(pass) to (a) a more
normalized trade-outcome distribution (H_GIVEBACK, session 23: ~43% win rate, pooled
median give-back 119%, Gini 0.58 among winners — edge carried by a minority of monster
winners) and (b) faster resolution, stated explicitly by the user: "a strategy with 50%
p_pass that resolves in 10 days is useful; one with 50% p_pass that resolves in 40 days
provides no advantage" — speed matters as much as the pass fraction. Every mechanism
tried since H_GIVEBACK (H_TP_EXIT, H_POST_TP_REENTRY, H_POST_TP_SCORE_V2) used a FULL
exit. This is the first test of a PARTIAL scale-out: bank a fraction of size once a
trade reaches a fixed (at-entry, ATR-multiple) favorable-excursion trigger, and leave
the remainder running on the exact same, untouched ATR-2.5x trailing stop — the runner
itself is never cut, only downsized once.

New file `research/tsmom_partial_scaleout_variant.py`. `vol_target_returns_partial_scaleout`
is a parallel copy of `vol_target_returns` (bank_frac=0.0 reproduces it bar-for-bar,
checked); `extract_trades_partial` (bank_frac=0.0 reproduces H_GIVEBACK's own
`extract_trades` exactly, same trade counts and realized/MFE returns, checked) adds
per-trade tracking of the scale-out event. Grid: trigger ∈ {0.75, 1.0, 1.5, 2.0} × ATR,
bank_frac ∈ {25%, 50%, 75%}, reusing `tsmom_exit_variants.py`'s generic walk-forward /
barrier-sim machinery unmodified. Candidate selection stayed on the project's standard
train-only Sharpe (to avoid a new overfitting risk on more easily gamed distribution
metrics), but the full grid's OOS numbers are reported for every candidate, since this
round is explicitly exploratory.

**Pooled trade-level distribution shape, OOS test trades, n=200 (trade count is
identical across every candidate — scale-out changes how a trade resolves, not how many
occur):**

| candidate | win rate | giveback (median) | skew | excess kurtosis | Gini (winners) |
|---|---|---|---|---|---|
| baseline (no scale-out) | 45.5% | 111.7% | 3.61 | 18.31 | 0.61 |
| trig=1.0×ATR, bank=25% | 49.5% | 101.2% | 3.21 | 15.51 | 0.60 |
| trig=1.5×ATR, bank=50% | 58.5% | 83.0% | 2.09 | 8.47 | 0.52 |
| trig=1.0×ATR, bank=75% | 75.0% | 78.7% | 0.45 | 2.80 | 0.47 |
| trig=2.0×ATR, bank=75% | 57.0% | 75.4% | 0.49 | **0.53** | 0.34 |

**This mechanism genuinely works at normalizing the distribution.** At bank=75%, win
rate crosses 50% for the first time in this project's history (57-75% vs. baseline's
45.5%), skew collapses from a heavily right-tailed 3.6 to near-symmetric 0.4-0.5,
excess kurtosis falls from 18.3 (extreme fat tail) to 0.5-3.8 (close to normal), and
winning-trade concentration drops (Gini 0.61 → 0.34-0.47). Give-back falls from 111.7%
to 75-85%. Every one of these moves monotonically with bank_frac.

**But it directly fails the user's own "acceleration" test.** Per-combo median days to
resolution roughly DOUBLES to TRIPLES at bank=75% (e.g. BTC 4h 16→34-37 days, ETH 12h
25→44-70 days, SOL 12h 24→46-59 days), and P(pass) drops in most combos (BTC 4h 24.4%→
17.6-24.8%, ETH 8h 28.8%→22.8-24.8%, ETH 12h 32.4%→22.4-25.6%, SOL 12h 25.2%→19.6-23.6%).
**This is structural, not a backtest artifact**: Propr's barrier is a FIXED target
(+9%), each new trade's size is set fresh by vol-targeting (not scaled by accumulated
equity), and banking part of a position removes exactly the exposure that would have
kept compounding through the rest of a trend — so the same favorable move now
contributes less to closing the gap to +9%, on average taking longer regardless of how
much "nicer" each individual trade's outcome looks. Distribution-normalization and
speed-to-resolution pull in opposite directions under any early-profit-banking
mechanism against a fixed terminal target — the same tension H_TP_EXIT's full-TP
version hit, just less severely here because the runner isn't fully cut.

**Where the tension is mildest**: light scale-out (bank=25%, trigger 1.0-1.5×ATR) gets
a real but modest distribution improvement (win rate 49.5-59.5% across combos, skew cut
roughly in half, give-back down 10-15 points) at a much smaller speed cost (median days
~1.3-1.5x baseline vs. ~2-3x at bank=75%), and P(pass)/Sharpe are roughly neutral in
most combos. One standout: **BTC 4h at trig=1.0×ATR, bank=25-50% beats the baseline
outright** on test Sharpe (2.06-2.19 vs. 1.95) while modestly improving win rate
(45.1-56.3% vs. 40.8%) and give-back (91.3-103.8% vs. 126.3%) — the one candidate in
this grid with no clear tradeoff for that combo specifically (though its own P(pass)
and speed are still roughly flat-to-slightly-worse, not improved).

**Pooled-account follow-up** (`research/tsmom_partial_scaleout_pooled.py`, user-prompted:
"what does this mean for the full portfolio?"): the per-combo numbers above use each
combo's own FULL account (`tsmom_exit_variants.py`'s convention, for apples-to-apples
comparison with H_TP_EXIT/H_WIDE_TRAIL) — not the actual live structure, which is ONE
$100K account quarter-weighted across all 4 combos simultaneously (H_JOINT_PORTFOLIO's
machinery). Re-run there, the picture changes in an important way: **pooled P(pass)
goes UP with scale-out, not down** (81.0% baseline → 87.6% light / 94.6% moderate /
97.1% heavy) — smoothing each combo's fat right tail also reduces the POOLED account's
variance enough to avoid the -3% static floor more often, an effect invisible when each
combo is judged alone at full size. But median days-to-resolution still balloons (83 →
123 → 186 → **353** days at the heaviest setting) — confirming the per-combo speed
finding holds, and more so, at the portfolio level.

**The single number that combines both — $/month across repeated eval cycles (renewal
sim, same convention as H_JOINT_PORTFOLIO) — settles it decisively against every
scale-out candidate tested:**

| candidate | pooled P(pass) | median days | $/month (mean) | cycles/yr |
|---|---|---|---|---|
| baseline (no scale-out) | 81.0% | 83 | **$1,630** | 3.44 |
| light (trig=1.0×ATR, bank=25%) | 87.6% | 123 | $1,249 (-23%) | 2.41 |
| moderate (trig=1.5×ATR, bank=50%) | 94.6% | 186 | $987 (-39%) | 1.76 |
| heavy (trig=1.0×ATR, bank=75%) | 97.1% | 353 | $556 (-66%) | 0.96 |
| heavy-wide (trig=2.0×ATR, bank=75%) | 91.4% | 288 | $625 (-62%) | 1.17 |

Every scale-out candidate has a HIGHER pass probability than baseline, and every one
still makes LESS money per month, because the resolution slowdown (cycles/yr falling
3.4x at the heaviest setting) more than cancels out the higher pass rate and the
smoother-looking trade distribution. This is exactly the failure mode the user's own
"resolves in 40 days provides no advantage" framing anticipated, now confirmed
quantitatively rather than just qualitatively.

**Revised verdict: the mechanism delivers a real, tunable, genuinely more normalized
per-trade return distribution (win rate, skew, kurtosis, give-back all improve
monotonically with bank_frac) — but on the metric that combines pass probability and
speed into the thing that actually matters (cash extracted per unit time), it makes the
portfolio strictly worse, and increasingly so the more aggressively it banks.** If
distribution shape is valued for its own sake (smoother equity curve, less reliance on
lucky monster trades) independent of $/month, `light (trig=1.0×ATR, bank=25%)` is the
least costly way to get a meaningful step in that direction (-23% $/month for a
real, if modest, improvement in win rate/skew/give-back). If $/month is what matters,
none of these beat the current production signal. No change made to the production
signal or locked config. Full results in `research/output/
tsmom_partial_scaleout_{per_combo,pooled,pooled_account}.json`.

### H_LEVERAGE_SPEED -- trading pass-probability for calendar-time-at-risk (session 28
continued, second user reframe)

**Motivation**: the dollars-per-month framing above assumes indefinite reinvestment and
treats calendar time as costless except through compounding rate. The user's actual
concern is different: the baseline's ~83-day median cycle means (a) up to ~83 days of
exposure to market-regime change silently breaking the strategy before a single cycle
even resolves, and (b) up to ~83 days of exposure per cycle to non-strategy tail risk
(platform/counterparty, security, regulatory). The ask, explicitly: accept a LOWER
p_pass in exchange for a variant that resolves in 7-30 days, so multiple independent
attempts fit inside a short, bounded calendar window instead of one long one -- the
opposite direction from H_PARTIAL_SCALEOUT (which only makes cycles slower).

**Method**: leverage is the natural dial for this, orthogonal to the scale-out
mechanism -- scaling the same per-combo OOS daily-return series up moves both the +9%
target and the -3%/daily floors closer in calendar-time terms (same MI&A-style overlay
concept already established in this project, session 10-14, applied here as a linear
rescale of the already-computed daily series -- consistent with how every pooled script
here treats a leverage multiplier). New file `research/tsmom_leverage_speed_sweep.py`.
Headroom check: LOCKED k's (0.437-0.488) sit well below the raw per-asset caps
(BTC/ETH 5x, SOL 2x -- cap/locked-k headroom ranges ~4.6x for SOL, ~10x for BTC/ETH), so
every k_extra tested here (up to 4x) keeps the effective multiplier under cap for all 4
combos (SOL reaches 1.75 vs. its 2.0 cap at k_extra=4, the tightest). **Caveat, stated
plainly**: this is a linear rescale of an already-realized return series, not a re-run
of the signal with per-bar cap enforcement -- some individual bars during low-realized-
vol periods could, in live execution, get silently clipped at the raw cap in a way this
backtest doesn't model. Treat this as a plausibility check on the dial's shape, not a
verified deployable design.

**Result: leverage does what scale-out could not -- it buys speed while also making
MORE money per month, not less, because the resolution speed-up outweighs the lower
per-attempt pass rate:**

| k_extra | P(pass) | median days | cycles/yr | $/month |
|---|---|---|---|---|
| 1.0 (current) | 81.0% | 83 | 3.44 | $1,630 |
| 1.5 | 62.3% | 36 | 7.13 | $2,542 |
| 2.0 | 54.3% | 22 | 10.26 | $3,156 |
| 3.0 | 48.1% | 16 | 15.31 | $4,070 |
| 4.0 | 36.6% | 11 | 21.57 | $4,229 |

P(resolved within T days) rises sharply with leverage: at k_extra=4.0, 41% of cycles
resolve within a week and 83% within a month, vs. 2.2%/13.2% at today's leverage --
directly answering the "can I get an answer inside a short window" question.

**Important nuance -- the sweet spot for a SHORT, FIXED window is not the highest
leverage tested.** P(pass) specifically within 30 days peaks around k_extra=2-3
(29.7%/32.2%) and is already declining by k_extra=4 (27.8%; within 60 days: 43.6%/43.1%
peak at k_extra=2-3, falling to 34.6% at k_extra=4) -- past some point, extra leverage
speeds up failures as much as passes, so cranking leverage further stops helping (and
starts hurting) the specific "pass inside T days" question, even though median speed
and dollars/month keep improving through k_extra=4 in this grid.

**Caveats**: (1) failure frequency in absolute terms rises a lot (19.1% to 51.2% of
cycles fail at k_extra=4, each costing the $450 eval fee) -- the EV is still better,
but this means many more individual losing outcomes along the way, a real
practical/psychological cost distinct from the EV math. (2) This tests the SAME
historical edge at higher size -- it doesn't itself protect against genuine regime
decay, but it does let a live deployment detect decay (via a failed short cycle) and
react far faster than an 83-day cycle would allow, which is the actual mechanism
connecting "speed" to the user's regime-risk concern. (3) Per-bar leverage-cap
enforcement not verified (see Method caveat above) -- recommended before treating any
specific k_extra as deployable.

No change made to the production signal, sizing, or locked config -- this is a
plausibility sweep, not a proposed deployment change. Full grid in
`research/output/tsmom_leverage_speed_sweep.json`.

**Per-bar cap verification** (`research/tsmom_leverage_speed_capped.py`, user-requested,
narrowed to the 1.5-3.0x region): re-ran the signal properly, applying
min(min(raw vol-target size, CAP) * locked_k * k_extra, CAP) bar-by-bar instead of
linearly rescaling the already-realized return series. Sanity check passed (k_extra=1.0
reproduces `build_daily_series()` -- today's production baseline -- exactly, for all 4
combos). **Result: zero clipping at every combo, at every k_extra from 1.5 through
3.0** -- the naive linear-rescale numbers above are exact in this range, not just a
reasonable approximation. The Method-section caveat about unverified per-bar clipping is
resolved for the 1.5-3x region specifically; it was never wrong there, just unverified
until now.

**Real-budget sequential-attempt follow-up** (`research/tsmom_leverage_ruin_sim.py`,
reusing session 22's `budget_ruin_sim.py` machinery unmodified against the user's real
~$637/590EUR budget): session 22 already established that firing multiple accounts in
PARALLEL gives near-zero pass/fail diversification (~1 correlation, same market driving
all of them at once) -- real diversification needs staggered starts, which the
sequential re-buy-on-fail simulation already models correctly. Rerunning it at $5K/$10K
Turbo tiers with k_extra=2.0-3.0 instead of baseline:

| k_extra | median days to first pass | p95 days (worst case) | mean fails before pass | P(ruin) |
|---|---|---|---|---|
| 1.0 (baseline) | 105 | 338 | 0.26 | 0.0% |
| 2.0 | 47 | 179 | 0.84 | 0.0% |
| 2.5 | 40 | 159 | 0.97 | 0.0% |
| 3.0 | 36 | 138 | 1.08 | 0.0% |

Ruin risk is a non-issue at every setting (confirms session 22's original finding, now
for the leverage variants too) -- $637 covers far more failed attempts than are ever
plausibly needed (mean spend including the eventual win: $31-52 at $5K, $63-104 at
$10K). The real, decisive effect is calendar time: k_extra=3.0 cuts median time to a
first real pass roughly 3x (105 to 36 days) and cuts the WORST-CASE (p95) tail even
more in absolute terms (338 to 138 days) -- directly answering the user's stated
concern about long unresolved cycles exposing the strategy to regime change and
calendar-time platform risk, at essentially zero added financial risk given the budget
available.

**Practical implication**: for the "get funded once, safely, fast" goal specifically,
sequential small-tier attempts (not parallel/simultaneous ones -- those don't
diversify) at k_extra≈2.5-3 dominate running baseline leverage on repeated attempts.
Once an account actually passes, the calculus changes: the funded account itself is now
the valuable asset to protect, not a disposable eval fee, so switching that specific
account back toward baseline (or lower) leverage is the natural choice for the
funded/compounding phase -- this was not re-simulated this session and is a direct
carryover of the same logic, not a new backtested claim.

---

