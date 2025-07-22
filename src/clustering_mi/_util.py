# Small helper functions
from __future__ import annotations

from collections.abc import Callable
from math import lgamma

import numpy as np
from numpy.typing import ArrayLike


def _log_factorial(n: ArrayLike | float) -> ArrayLike | float:
    """
    Compute the natural logarithm of the factorial of n.

    Parameters
    ----------
    n : ArrayLike | float
        The value for which to compute the log-factorial.

    Returns
    -------
    ArrayLike | float
        log(n!) (base e)
    """
    if isinstance(n, (list, np.ndarray)):
        return np.array([lgamma(x + 1) for x in n])
    return lgamma(n + 1)


def _log_binom(n: float, m: float) -> float:
    """
    Compute the natural logarithm of the binomial coefficient binomial(n, m).

    Parameters
    ----------
    n : float
        The number of items.
    m : float
        The number of items to choose.

    Returns
    -------
    float
        log(binomial(n, m)) (base e)
    """
    return _log_factorial(n) - _log_factorial(m) - _log_factorial(n - m)


def _log_Omega_EC(
    rs: ArrayLike,
    cs: ArrayLike,
    useShortDimension: bool = False,
    symmetrize: bool = False,
) -> float:
    """
    Approximate the log of the number of contingency tables with given row and column sums
    using the EC estimate of Jerdee, Kirkley, Newman (2022) https://arxiv.org/abs/2209.14869.

    Parameters
    ----------
    rs : list or int
        Row sums.
    cs : list or int
        Column sums.
    useShortDimension : bool, optional
        Whether to optimize the encoding by possibly swapping the definitions of rows and columns (default: True).
    symmetrize : bool, optional
        Whether to symmetrize the estimate (default: False).

    Returns
    -------
    float
        Estimate of log_Omega (base 2).
    """
    rs = np.array(rs)
    cs = np.array(cs)
    # Remove any zeros
    rs = rs[rs > 0]
    cs = cs[cs > 0]
    if len(rs) == 0 or len(cs) == 0:
        return -np.inf  # There are no tables
    if (
        useShortDimension
    ):  # Performance of the EC estimate is generally improved when there are
        # more rows than columns. If this is not the case, swap definitions around
        if len(rs) >= len(cs):
            return _log_Omega_EC(rs, cs, useShortDimension=False)
        return _log_Omega_EC(cs, rs, useShortDimension=False)
    if symmetrize:
        return (
            _log_Omega_EC(rs, cs, symmetrize=False)
            + _log_Omega_EC(cs, rs, symmetrize=False)
        ) / 2
    m = len(rs)
    N: float = np.sum(rs)
    if (
        len(cs) == N
    ):  # In this case, we may simply return the exact result (equivalent to alpha = inf)
        return _log_factorial(N + 1) - np.sum(_log_factorial(rs + 1))
    alphaC = (N**2 - N + (N**2 - np.sum(cs**2)) / m) / (np.sum(cs**2) - N)
    result = -_log_binom(N + m * alphaC - 1, m * alphaC - 1)
    for r in rs:
        result += _log_binom(r + alphaC - 1, alphaC - 1)
    for c in cs:
        result += _log_binom(c + m - 1, m - 1)
    return float(result / np.log(2))  # Convert to base 2


# Only including this due to issues with scipy.optimize.minimize_scalar
def _minimize_golden_section_log(
    f: Callable[[float], float], min_val: float, max_val: float, tol: float = 1e-5
) -> tuple[float, float]:
    """
    Optimize a function using the golden section search method. Optimization occurs in the logarithmic domain.

    Parameters
    ----------
    f : callable
        The function to optimize.
    min_val : float
        The lower bound of the search interval. Must be greater than 0.
    max_val : float
        The upper bound of the search interval. Must be greater than 0.
    tol : float, optional
        The tolerance for convergence in the log(parameter) (default: 1e-5).

    Returns
    -------
    tuple[float, float]
        The minimizing value of the parameter and the function value at that point.
    """
    a = np.log(min_val)
    b = np.log(max_val)

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    resphi = 2 - phi

    c = a + resphi * (b - a)
    d = b - resphi * (b - a)

    while abs(c - d) > tol:
        if f(np.exp(c)) < f(np.exp(d)):
            b = d
            d = c
            c = a + resphi * (b - a)
        else:
            a = c
            c = d
            d = b - resphi * (b - a)

    return np.exp((a + b) / 2), f(np.exp((a + b) / 2))
