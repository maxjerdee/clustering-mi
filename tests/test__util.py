from __future__ import annotations

import numpy as np
import pytest

from clustering_mi._util import (
    _log_binom,
    _log_factorial,
    _log_Omega_EC,
    _minimize_golden_section_log,
)


def test__log_factorial():
    n = 5
    result = 1
    for i in range(1, n + 1, 1):
        result *= i
    assert _log_factorial(n) == pytest.approx(np.log(result))

    ns = [3, 4]
    results = [np.log(6), np.log(24)]
    for n, expected in zip(ns, results):
        assert _log_factorial(n) == pytest.approx(expected)


def test__log_binom():
    n = 5
    m = 2
    assert _log_binom(n, m) == pytest.approx(np.log(10))


def test__log_Omega_EC():
    rs = [1, 3, 4, 4]
    cs = [5, 2, 2, 3]
    result = 9.4314
    assert _log_Omega_EC(rs, cs) == pytest.approx(result)


def test__minimize_golden_section_log():
    f = lambda x: x**2 - 3 * x
    x_true = 1.5
    f_val_true = -2.25

    x, f_val = _minimize_golden_section_log(f, 0.01, 100)

    assert x == pytest.approx(x_true, rel=1e-2)
    assert f_val == pytest.approx(f_val_true, rel=1e-2)
