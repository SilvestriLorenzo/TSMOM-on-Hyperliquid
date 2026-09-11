# TSMOM-on-Hyperliquid
A vol-targeted time-series momentum (TSMOM) strategy for BTC/ETH/SOL perpetuals on Hyperliquid, built to pass and run on [Propr](https://propr.xyz) funding challenges.

This repo is the research trail: every hypothesis tested, including the ones that
failed, with the validation discipline applied throughout (walk-forward splits,
block-bootstrap Monte Carlo, causal-timing checks, outlier-robustness checks, explicit
sanity assertions before trusting any result). It is currently running on Propr's paper
(Beta) environment for live-execution validation before real capital goes in.

## TL;DR

- **Signal**: sign of the most recent bar's return (classic TSMOM), at 4h/8h/12h bar
  resolution, vol-targeted position sizing, ATR-14/2.5x trailing stop, no target/no cap
  on the winning side.
- **Validated out-of-sample**: genuine chronological 70/30 walk-forward split (parameter
  selection and combo selection both restricted to the training window), block-bootstrap
  Monte Carlo (500-2000 paths) checked against Propr's real evaluation barriers (+9%
  target, -3% static drawdown, -3% daily loss).
- **Headline result**: running the 4 locked combos (BTC 4h, ETH 8h, ETH 12h, SOL 12h)
  together on one pooled account (as Propr's rules actually require, not each backtested
  in isolation) gives **P(pass) = 81.0%** full-sample / **62.2%** ex the one clear rally
  episode in the data, Sharpe 2.07 / 1.09, a real diversification benefit from
  imperfectly-correlated combos (0.50-0.75 pairwise correlation).
- **Deployed** to Propr's Beta (paper) environment, executing real orders against real
  Hyperliquid market data with simulated capital — the closest thing to production
  short of real money.

## The strategy

**Entry**: at the start of each bar, if flat, check the sign of the prior bar's return.
If defined (nonzero, valid trailing vol/ATR), enter in that direction.

**Sizing**: vol-targeted — position size = min(target_vol_per_bar / trailing 20-bar
realized vol, per-asset leverage cap), fixed at entry, scaled by a locked leverage
multiplier `k` calibrated via Magdon-Ismail & Atiya's expected-max-drawdown formula
against Propr's own 3% drawdown budget.

**Exit**: ATR-14 trailing stop at 2.5x, ratcheting favorably only (never loosens). No
take-profit — every attempt to cap the winning side (see below) made things worse.

**Locked deployment config** (`research/tsmom_deployment_config.py`,
`research/tsmom_joint_portfolio_barrier_sim.py`):

| Combo | lookback (bars) | leverage k | OOS P(pass), standalone |
|---|---|---|---|
| BTC 4h | 1 | 0.479 | 32.4% |
| **ETH 8h** | 1 | **0.488** | **36.0%** |
| ETH 12h | 1 | 0.479 | 33.0% |
| SOL 12h | 2 | 0.437 | 29.2% |

Run together, quarter-weighted on one pooled account (matching how the live executor
actually allocates capital) — **P(pass) 81.0%**, well above any single combo alone,
because the three assets' momentum states are correlated but not identical (0.50-0.75
pairwise daily correlation), so the pooled account's variance is genuinely lower than
"4x one combo" would imply.

## Validation methodology

The thing this project treats as non-negotiable: every number here comes with a check
for the specific way it could be wrong.

- **Walk-forward, not in-sample**: lookback parameter and which coin/interval "looks
  best" are both chosen on a training slice only; every reported Sharpe/P(pass) is
  computed on the held-out test slice. The original in-sample headline numbers were
  *higher* than the walk-forward numbers turned out to be for some combos — a red flag
  the project didn't shy away from reporting.
- **Path-dependent Monte Carlo, not a closed-form approximation**: P(pass) is estimated
  by block-bootstrapping real market data (preserving local trend/volatility
  clustering), reconstructing a synthetic OHLC path, and re-running the entry/exit
  signal fresh on it — not resampling the strategy's own historical returns, which would
  inherit one specific realized path.
- **Causal-timing discipline, enforced explicitly**: every stop/entry check uses data
  through bar t-1 only, checked against bar t's own intrabar high/low. A real look-ahead
  bug was found and fixed early on (a sizing overlay evaluated with full-sample
  information) — it's referenced throughout the research log as the reason every
  subsequent script gets an explicit "does this look ahead?" pass before its results are
  trusted.
- **Outlier-robustness checks on every headline number**: drop the 20 largest bars (or
  10 largest trades) and see if the result survives. TSMOM does; several other
  hypotheses in this project's broader search did not (see below).
- **Real transaction costs**, checked against actual Hyperliquid L2 order-book depth,
  not just a flat assumption (the flat 9bps round-trip convention was validated to move
  results by <0.5pp — cheap to check, worth checking).
- **A trade-level give-back diagnostic**, because "the Sharpe looks fine" can still hide
  an ugly trade-by-trade distribution: pooled median give-back ratio of peak unrealized
  gain is 119% (the median trade that's ever green still closes a net loser), win rate
  ~43-45%, driven by a real but not extreme concentration in the largest winners (Gini
  0.58). This is standard trend-following payoff shape, confirmed directly rather than
  assumed.

## Iterating on the locked strategy

Several follow-on ideas were tested and honestly reported, whether they helped or not:

- **Directional restriction** (long-only/short-only, to reduce the three assets'
  positions "cancelling out"): hurts pooled P(pass) — restricting direction *raises*
  cross-combo correlation (0.70-0.95 vs. 0.50-0.75), which is the opposite of the
  intuition behind trying it.
- **Capped take-profit / wider trailing stop**: no combo shows a robust improvement over
  the locked ATR-2.5x stop; a capped TP is a clean kill (uniformly worse OOS Sharpe).
- **Two-stage post-TP re-entry/scoring mechanisms**: modest, inconsistent gains — a
  holdout window was deliberately locked and left untouched rather than kept tuning
  against the same test slice indefinitely (this project treats repeated-look leakage
  across research sessions as a real risk, not just single-run overfitting).
- **A cross-asset leader signal** (BTC's momentum driving ETH/SOL's direction, instead
  of each computing its own): hurts pooled P(pass) for the same correlation-cost reason
  as directional restriction — a second, mechanism-distinct confirmation of the same
  finding.
- **Partial profit-taking** (bank part of a winning position at a fixed favorable-
  excursion trigger, let the rest ride on the same unmodified stop): genuinely
  normalizes the trade-return distribution — win rate crosses 50% for the first time,
  skew/kurtosis collapse toward normal — but it does this by slowing the path to
  Propr's fixed profit target, making it strictly worse on a combined pass-probability
  and speed basis. A clean, quantified example of a real tradeoff rather than a free
  lunch.
- **Leverage as a speed lever** (the current frontier): scaling position size up
  (verified clean against per-bar leverage caps) trades some P(pass) for a much faster
  path to resolution — e.g. 3x leverage cuts median calendar days-to-a-first-pass from
  ~105 to ~36 against a small real budget, with negligible ruin risk, and a genuine
  "sweet spot" leverage level for maximizing P(pass) *within* a short fixed window
  rather than just maximizing P(pass) unconditionally. Currently running live in
  parallel with the baseline signal on Propr Beta (three paper accounts: baseline, 2x,
  3x leverage) to validate the mechanism end-to-end before committing real capital.


TSMOM is the strategy that survived out of a broader search, not the only thing tried.
A spot-BTC-ETF-flow-conditioned long bias (`H_ETF`) was the most promising alternative:
a genuine walk-forward Sharpe of 1.52 (t=1.77, just short of conventional significance)
that clearly beat a same-window no-signal baseline (P(pass) 28.4% vs. 12.1%) — real,
but it didn't extend to ETH and didn't clear the same robustness bar TSMOM did, so it
wasn't pursued to deployment. An order-flow-imbalance effect on BTC was confirmed but
economically too small to trade; a matching pattern on SOL remains genuinely open,
limited by how much trade-tape data had been collected at the time. Several other
directions — BTC/ETH/SOL cointegration/lead-lag, a ported impulse-continuation mechanic,
a TWAP-detection heuristic — were tested with the same rigor and killed outright once
checked out-of-sample or against ground truth, which is itself the point: this project's
default assumption for any new idea is that it's wrong until a walk-forward, cost-aware,
outlier-checked test says otherwise.

## Repo structure

```
research/
  tsmom_hypothesis.py                base signal, naive + ATR-trailing-stop backtest
  tsmom_walkforward.py               70/30 walk-forward harness (the OOS backbone)
  tsmom_barrier_sim.py               block-bootstrap Monte Carlo vs. Propr's barriers
  mi_atiya_drawdown.py               Magdon-Ismail & Atiya expected-max-drawdown sizing
  tsmom_sizing_retune.py             sizing-overlay retune after walk-forward
  tsmom_deployment_config.py         selects the final locked (coin, interval, lb, k) set
  tsmom_joint_portfolio_barrier_sim.py   pooled 4-combo account simulation (the real structure)
  tsmom_directional_variants.py      long/short-restriction test
  tsmom_exit_variants.py             capped-TP / wider-trail exit variants
  tsmom_giveback_diagnostic.py       trade-level give-back / concentration diagnostic
  tsmom_post_tp_reentry.py, tsmom_post_tp_score_v2.py   two-stage post-TP mechanisms
  tsmom_btc_leader_variant.py        cross-asset leader-signal variant
  tsmom_partial_scaleout_variant.py, tsmom_partial_scaleout_pooled.py   partial profit-taking
  tsmom_leverage_speed_sweep.py, tsmom_leverage_speed_capped.py   leverage-as-speed-lever
  tsmom_leverage_ruin_sim.py, tsmom_funded_phase_switch_sim.py   real-budget/funded-phase economics
  budget_ruin_sim.py, sharpe_se_analysis.py   business-decision math
  tsmom_gap_hypothesis.py, tsmom_fast_hypothesis.py, tsmom_funded_phase_sim.py
  fetch_reference_data.py, fetch_tsmom_candles.py   data pull (Hyperliquid public API)
  output/                            candle data (BTC/ETH/SOL x 4h/8h/12h) + every result JSON
RESEARCH_LOG.md                      not uploaded but avaiable
```

**Running**: most scripts are self-contained — `python research/tsmom_walkforward.py`
from the repo root. Two (`fetch_tsmom_candles.py`, `sharpe_se_analysis.py`) use
package-style imports and need `python -m research.<name>` from the repo root instead.
Candle data is already included in `research/output/`, so nothing needs to be re-fetched
to reproduce any result. Dependencies: `numpy`, `pandas`, `httpx` (only needed for the
fetch scripts). Every script's `__main__` block runs its own sanity checks first (e.g.
"does this parallel-copied function reproduce the validated one bit-for-bit") and
refuses to print results if they fail — that discipline is part of the deliverable, not
just the numbers.

## Honest bottom line

- The pooled-account correlation structure and Monte Carlo are estimated from a common
  overlap window of ~250 days across all four combos — real, but short; a longer live
  track record is the actual confirmation, not another backtest.
- Every P(pass) number assumes the historically-estimated return distribution holds
  going forward. Nothing here claims the edge can't decay — a holdout window was
  deliberately locked and left untouched specifically so a genuine out-of-sample
  confirmation is still possible later, rather than quietly re-tuning against the same
  data indefinitely.
- The live Beta deployment code (order execution, account state, credentials) isn't in
  this repo — this is the research trail, not the production trading bot.
