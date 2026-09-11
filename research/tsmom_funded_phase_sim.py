"""
H_TSMOM funded-phase renewal simulation (session 15), continuing
tsmom_deployment_config.py. That script answers "does ONE evaluation pass, fail, or
stall on the way to +9%/-3%" -- this answers the user's actual next question: "if it
passes, and I keep going, how much money per month, and does that change if I scale up
account size."

MODELING ASSUMPTION, not confirmed in PROPR.md, flagged explicitly: Propr's docs say
passing a "Turbo 1-Step" evaluation gets you "access to funded capital" and thereafter
"keep 80% of the gains," with a full-sweep-on-request mechanic that resets the balance
to the starting amount and re-arms the risk limits from there (PROPR.md SS4/SS8). What
is NOT stated: whether the evaluation's own +9% gain itself becomes the first payout
(the "1-step" naming, and every real-money 1-step prop-firm model this project's author
is aware of, suggests yes), or whether a second +9% cycle is required inside the funded
account before anything is withdrawable. This sim assumes the former (every completed
+9%/-3% cycle, evaluation or post-funding, is mechanically identical and immediately
sweepable) -- VERIFY against Propr's actual dashboard/T&Cs before relying on this
financially; if a second confirmation cycle is actually required, treat every number
below as roughly one extra cycle's delay from being right, not fundamentally different.

Policy modeled: sweep immediately after every pass (banks the $ real and permanently;
resets that account to its starting balance and re-plays an IDENTICAL fresh cycle at the
same locked leverage -- no re-purchase needed, since sweeping doesn't revoke the funded
account). On a fail (breach static or daily-loss floor), the funded account is assumed
revoked -- re-entry requires buying a fresh evaluation at the same tier's fee. This is a
renewal process: geometric number of passes accumulate between fails, each fail costs an
eval-fee restart. Uses the exact same simulate_path() as tsmom_deployment_config.py's
locked configs -- same leverage k, same bar_range (test-slice only, walk-forward-honest),
same block-bootstrap source -- just called repeatedly instead of once.

Per-dollar framing: p_pass, cycle length, and the 80%/9% payout math don't depend on
account size, and the challenge fee is ~0.45-0.50% of size at every tier (PROPR.md SS2) --
so $/month scales close to linearly with capital deployed. Everything here is computed
per $25K account and reported as both a dollar figure and a $-per-$1000-deployed rate,
so scaling to $50K/$100K is a straightforward multiply, not a separate simulation.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsmom_walkforward import walk_forward_one
from tsmom_barrier_sim import simulate_path, extract_market_tuples, BARS_PER_DAY

TARGET = 0.09
SPLIT = 0.80
ACCOUNT_SIZE = 25_000
EVAL_FEE = 125  # Turbo 1-Step, $25K tier, PROPR.md SS2
CYCLE_CAP_DAYS = 1095  # safety cap per cycle -- matches the horizon already shown to resolve almost all paths
SIM_YEARS = 3
N_SIMS = 1000

LOCKED = {
    ("BTC", "4h"): 0.479, ("ETH", "8h"): 0.488, ("ETH", "12h"): 0.479, ("SOL", "12h"): 0.437,
}


def simulate_renewal(coin: str, interval: str, lb_star: int, bar_range: tuple[int, int],
                      k: float, total_days: float, rng: np.random.Generator) -> dict:
    log_ret_hist, hi_hist, lo_hist = extract_market_tuples(coin, interval, bar_range)
    bars_per_day = BARS_PER_DAY[interval]
    cycle_cap_bars = int(CYCLE_CAP_DAYS * 24 / int(interval[:-1]))
    days_elapsed, net_cash, n_pass, n_fail = 0.0, 0.0, 0, 0
    while days_elapsed < total_days:
        outcome, bars = simulate_path(coin, interval, lb_star, 14, 2.5, k, False, cycle_cap_bars,
                                       log_ret_hist, hi_hist, lo_hist, rng, return_bars=True)
        days_elapsed += bars / bars_per_day
        if outcome == "pass":
            net_cash += ACCOUNT_SIZE * TARGET * SPLIT
            n_pass += 1
        elif outcome in ("fail_static", "fail_daily"):
            net_cash -= EVAL_FEE
            n_fail += 1
        else:  # unresolved inside the cycle cap -- rare (see module docstring), stop the clock here
            break
    return {"net_cash": net_cash, "n_pass": n_pass, "n_fail": n_fail, "days": days_elapsed}


if __name__ == "__main__":
    total_days = SIM_YEARS * 365
    print(f"Renewal simulation: sweep-immediately policy, {N_SIMS} runs x {SIM_YEARS}y each, "
          f"$25K account, {EVAL_FEE=}, {TARGET=:.0%}, {SPLIT=:.0%}\n")
    print(f"{'coin':>4} {'interval':>8} | {'$/mo mean':>10} | {'$/mo median':>12} | "
          f"{'p5':>8} | {'p95':>8} | {'P(net<0)':>9} | {'passes/yr':>9} | {'fails/yr':>9}")

    for (coin, interval), k in LOCKED.items():
        wf = walk_forward_one(coin, interval)
        rng = np.random.default_rng(hash((coin, interval)) % (2**32))
        results = [simulate_renewal(coin, interval, wf["lb_star"], wf["bar_range"], k, total_days, rng)
                   for _ in range(N_SIMS)]
        per_month = np.array([r["net_cash"] / (r["days"] / 30.44) for r in results])
        passes_per_yr = np.mean([r["n_pass"] / (r["days"] / 365) for r in results])
        fails_per_yr = np.mean([r["n_fail"] / (r["days"] / 365) for r in results])
        print(f"{coin:>4} {interval:>8} | {per_month.mean():>9.0f}$ | {np.median(per_month):>11.0f}$ | "
              f"{np.percentile(per_month, 5):>7.0f}$ | {np.percentile(per_month, 95):>7.0f}$ | "
              f"{np.mean(per_month < 0):>8.1%} | {passes_per_yr:>9.2f} | {fails_per_yr:>9.2f}")

    print(f"\nPer-$1000-deployed monthly rate = the $/mo mean above / 25 (since ACCOUNT_SIZE=$25K) --"
          f" multiply by (account size deployed / 1000) to rescale to $50K/$100K/combined.")
