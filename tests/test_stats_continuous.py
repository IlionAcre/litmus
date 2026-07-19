import numpy as np
import pytest

from litmus.stats import bootstrap_diff_ci, mann_whitney_test


def test_bootstrap_detects_a_real_shift():
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=100, scale=10, size=200).tolist()
    candidate = rng.normal(loc=130, scale=10, size=200).tolist()

    result = bootstrap_diff_ci(baseline, candidate, n_resamples=2000, random_state=42)

    assert result.significant is True
    assert result.observed_diff == pytest.approx(30, abs=5)
    assert result.ci_low > 0


def test_bootstrap_does_not_flag_pure_noise():
    rng = np.random.default_rng(7)
    baseline = rng.normal(loc=100, scale=10, size=200).tolist()
    candidate = rng.normal(loc=100, scale=10, size=200).tolist()

    result = bootstrap_diff_ci(baseline, candidate, n_resamples=2000, random_state=7)

    assert result.significant is False
    assert result.ci_low <= 0 <= result.ci_high


def test_mann_whitney_detects_a_real_shift():
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=100, scale=10, size=200).tolist()
    candidate = rng.normal(loc=150, scale=10, size=200).tolist()

    result = mann_whitney_test(baseline, candidate)

    assert result.significant is True
    assert result.p_value < 0.05


def test_mann_whitney_does_not_flag_pure_noise():
    rng = np.random.default_rng(7)
    baseline = rng.normal(loc=100, scale=10, size=200).tolist()
    candidate = rng.normal(loc=100, scale=10, size=200).tolist()

    result = mann_whitney_test(baseline, candidate)

    assert result.significant is False
    assert result.p_value >= 0.05
