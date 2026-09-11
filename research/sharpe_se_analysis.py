"""
H_SHARPE_SE -- how much live/paper sample a "solid" rolling Sharpe actually needs
(session 22, user-prompted: "how big the sample need to be to have a solid rolling
sharpe?"). Answers this with the standard asymptotic result (Lo, 2002, "The Statistics
of Sharpe Ratios") rather than a rule of thumb.

For i.i.d. per-period returns with per-period Sharpe s, the sample Sharpe estimator has
asymptotic variance Var(s_hat) = (1 + 0.5*s^2)/n, n = number of periods. Annualizing by
sqrt(q) periods/year (q=365 for daily) and substituting n = q*T (T = years of history)
gives, after simplification:

    SE(SR_hat) = sqrt((1 + SR^2/(2*q)) / T)

which for daily data and SR in the 0-3 range collapses to SE(SR_hat) ~= 1/sqrt(T) -- the
correction term is a rounding error at daily frequency. This is a LOWER BOUND on the
true SE: it assumes i.i.d. returns, while TSMOM positions are held ~10 bars on average
(real positive serial correlation), so the honest SE is somewhat worse than what's
computed here, not better.
"""
import numpy as np

Q = 365  # daily resampling, matches tsmom_joint_portfolio_barrier_sim.py's convention

# Measured Sharpes this project has actually produced (not hypothetical inputs):
SHARPES = {
    "Pooled 4-combo, ex Aug 17-27 rally (H_JOINT_PORTFOLIO, the honest number)": 1.11,
    "Pooled 4-combo, full 250d sample (inflated by the rally episode)": 2.07,
    "Best solo combo: ETH 8h (H_TSMOM deployment config)": 1.24,
    "Weakest tested combo in the 27-cell grid (H_TSMOM)": 0.41,
}

HORIZONS_YEARS = {
    "1 week (current Beta run length)": 7 / 365,
    "90 days (this project's own proposed rolling-Sharpe check window)": 90 / 365,
    "6 months": 0.5,
    "1 year": 1.0,
    "2 years": 2.0,
    "4 years": 4.0,
    "9 years": 9.0,
}


def se_sharpe(sr: float, t_years: float, q: int = Q) -> float:
    return np.sqrt((1 + sr**2 / (2 * q)) / t_years)


if __name__ == "__main__":
    for label, sr in SHARPES.items():
        print(f"\n=== True Sharpe = {sr:.2f} ({label}) ===")
        print(f"{'horizon':<55} {'SE':>6} {'95% CI':>18} {'excludes 0?':>12}")
        for hlabel, t in HORIZONS_YEARS.items():
            se = se_sharpe(sr, t)
            lo, hi = sr - 1.96 * se, sr + 1.96 * se
            excludes_zero = "yes" if lo > 0 else "no"
            print(f"{hlabel:<55} {se:>6.2f} {f'[{lo:>5.2f}, {hi:>5.2f}]':>18} {excludes_zero:>12}")

    print("\nNote: this is a LOWER BOUND on the true SE -- assumes i.i.d. daily returns;")
    print("TSMOM's ~10-bar average hold introduces real positive serial correlation this")
    print("formula ignores, so actual estimation uncertainty is somewhat worse than shown.")
