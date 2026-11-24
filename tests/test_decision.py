import numpy as np
import pytest

from decision import (bootstrap_threshold_interval, empirical_threshold,
                      theoretical_threshold)


def test_theoretical_threshold_derives_from_costs():
    assert theoretical_threshold(5.0, 1.0) == pytest.approx(1 / 6)
    assert theoretical_threshold(9.0, 1.0) < theoretical_threshold(2.0, 1.0)


def test_empirical_threshold_moves_down_when_misses_cost_more():
    rng = np.random.default_rng(4)
    y = (rng.random(3000) < 0.3).astype(int)
    p = np.clip(0.55 * y + rng.normal(0.2, 0.18, len(y)), 0.001, 0.999)
    t2, _ = empirical_threshold(y, p, 2.0, 1.0)
    t9, _ = empirical_threshold(y, p, 9.0, 1.0)
    assert t9 <= t2


def test_bootstrap_threshold_interval_is_reproducible():
    y = np.array([0, 0, 0, 1, 1, 1] * 20)
    p = np.linspace(0.05, 0.95, len(y))
    a = bootstrap_threshold_interval(y, p, 5.0, 1.0, n_boot=50, seed=7)
    b = bootstrap_threshold_interval(y, p, 5.0, 1.0, n_boot=50, seed=7)
    assert a == b
