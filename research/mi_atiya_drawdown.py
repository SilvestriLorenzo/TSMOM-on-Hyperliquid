"""
Magdon-Ismail & Atiya (2004), "On the Maximum Drawdown of a Brownian Motion", J. Applied
Probability 41:147-161. Used here to calibrate a leverage multiplier that targets a
given expected maximum drawdown for the TSMOM strategy's sizing.

For X(t) = sigma*W(t) + mu*t (Brownian motion with drift), the expected maximum drawdown
over horizon T is, for mu > 0 (eq. 27):
    E[D] = (2*sigma^2/mu) * Qp(alpha^2),   alpha = mu*sqrt(T/(2*sigma^2))
where Qp is a function the paper can only express as a non-elementary integral (eq. 28) --
its own numerical evaluation is "computationally... straightforward" per the authors, but
they publish a reference table (Appendix B) precisely so downstream users don't have to
re-derive the integral. That table is reproduced here and interpolated (log-linearly in
x, since it spans several orders of magnitude); outside the tabulated range the paper's
own asymptotic forms take over (Qp(x) -> gamma*sqrt(2x) as x->0, -> 0.25*log(x)+0.49088 as
x->infinity).

Key simplification used throughout this project's application: scaling a process by a
leverage multiplier k (mu -> k*mu, sigma -> k*sigma) leaves the Sharpe ratio mu/sigma
unchanged, and alpha^2 = mu^2*T/(2*sigma^2) = (mu/sigma)^2 * T/2 depends only on that
ratio -- so alpha, and therefore Qp(alpha^2), is invariant under uniform leverage
scaling. Since E[D]_k = (2*(k*sigma)^2/(k*mu)) * Qp(alpha^2) = k * (2*sigma^2/mu) *
Qp(alpha^2) = k * E[D]_1, expected max drawdown scales EXACTLY linearly with leverage.
Calibrating "the leverage that targets an expected max drawdown of X" therefore needs no
root-finding: compute E[D] once at k=1, then k_target = X / E[D]_1.
"""
import numpy as np

GAMMA = np.sqrt(np.pi / 8)  # ~0.6267, the mu=0 constant

# Appendix B, Qp(x) column (mu > 0 case) -- (x, Qp(x)) pairs, verbatim from the paper.
_QP_TABLE = [
    (0.0005, 0.019690), (0.0010, 0.027694), (0.0015, 0.033789), (0.0020, 0.038896),
    (0.0025, 0.043372), (0.0050, 0.060721), (0.0075, 0.073808), (0.0100, 0.084693),
    (0.0125, 0.094171), (0.0150, 0.102651), (0.0175, 0.110375), (0.0200, 0.117503),
    (0.0225, 0.124142), (0.0250, 0.130374), (0.0275, 0.136259), (0.0300, 0.141842),
    (0.0325, 0.147162), (0.0350, 0.152249), (0.0375, 0.157127), (0.0400, 0.161817),
    (0.0425, 0.166337), (0.0450, 0.170702), (0.0500, 0.179015), (0.0600, 0.194248),
    (0.0700, 0.207999), (0.0800, 0.220581), (0.0900, 0.232212), (0.1000, 0.243050),
    (0.2000, 0.325071), (0.3000, 0.382016), (0.4000, 0.426452), (0.5000, 0.463159),
    (1.5000, 0.668992), (2.5000, 0.775976), (3.5000, 0.849298), (4.5000, 0.905305),
    (10.0000, 1.088998), (20.0000, 1.253794), (30.0000, 1.351794), (40.0000, 1.421860),
    (50.0000, 1.476457), (150.0000, 1.747485), (250.0000, 1.874323), (350.0000, 1.958037),
    (450.0000, 2.020630), (1000.0000, 2.219765), (2000.0000, 2.392826),
    (3000.0000, 2.494109), (4000.0000, 2.565985), (5000.0000, 2.621743),
]
_QP_X = np.array([p[0] for p in _QP_TABLE])
_QP_Y = np.array([p[1] for p in _QP_TABLE])
_QP_LOGX = np.log(_QP_X)


def Qp(x: float) -> float:
    """mu > 0 case. Table-interpolated (log-linear in x) within [5e-4, 5000], the
    paper's own asymptotics outside that range."""
    if x <= 0:
        return 0.0
    if x < _QP_X[0]:
        return GAMMA * np.sqrt(2 * x)
    if x > _QP_X[-1]:
        return 0.25 * np.log(x) + 0.49088
    return float(np.interp(np.log(x), _QP_LOGX, _QP_Y))


def expected_max_drawdown(mu: float, sigma: float, T: float) -> float:
    """E[max drawdown] over horizon T for X(t)=sigma*W(t)+mu*t, mu>0 case (eq. 27). mu,
    sigma, T must be in consistent units (e.g. all per-bar / in bar counts)."""
    if mu <= 0:
        raise ValueError("this project only calibrates the mu>0 case (a strategy with "
                          "real historical positive drift) -- mu<=0 has no meaningful "
                          "'target drawdown' leverage to solve for")
    alpha = mu * np.sqrt(T / (2 * sigma ** 2))
    return (2 * sigma ** 2 / mu) * Qp(alpha ** 2)


def calibrate_leverage_for_target_mdd(mu: float, sigma: float, T: float,
                                       target_mdd: float) -> float:
    """The leverage multiplier k such that a process with drift k*mu, vol k*sigma has
    E[max drawdown over T] == target_mdd. No root-finding needed -- see module
    docstring: E[MDD] scales exactly linearly with k."""
    mdd_at_unit_leverage = expected_max_drawdown(mu, sigma, T)
    return target_mdd / mdd_at_unit_leverage


if __name__ == "__main__":
    # Sanity check against the paper's own worked asymptotic: Qp(x) -> 0.25*log(x)+0.49088
    # as x -> infinity; confirm the table and the asymptotic formula agree at the top of
    # the tabulated range, where both are valid.
    x_check = 5000.0
    table_val = Qp(x_check)
    asymptotic_val = 0.25 * np.log(x_check) + 0.49088
    print(f"Qp({x_check}) table={table_val:.6f} vs asymptotic={asymptotic_val:.6f} "
          f"(should match closely at the top of the tabulated range)")

    x_check2 = 0.0005
    table_val2 = Qp(x_check2)
    small_x_val2 = GAMMA * np.sqrt(2 * x_check2)
    print(f"Qp({x_check2}) table={table_val2:.6f} vs small-x asymptotic={small_x_val2:.6f} "
          f"(should match at the bottom of the tabulated range)")

    # Worked example: a process with mu=0.001/bar, sigma=0.02/bar (roughly matching
    # H_TSMOM's per-bar scale), what's E[MDD] over a 1095-bar horizon, and what leverage
    # targets 1.5%?
    mu, sigma, T = 0.001, 0.02, 1095
    mdd = expected_max_drawdown(mu, sigma, T)
    k = calibrate_leverage_for_target_mdd(mu, sigma, T, target_mdd=0.015)
    print(f"\nExample: mu={mu}, sigma={sigma}, T={T} bars -> E[MDD]={mdd:.4f} "
          f"({mdd:.1%}); leverage for 1.5% target: k={k:.3f}")
