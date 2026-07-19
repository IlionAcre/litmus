import numpy as np
import pytest

from litmus.stats import bootstrap_diff_ci


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


def test_bootstrap_uses_pairing_not_just_independent_distributions():
    """A case where the *pairing* itself carries the signal: baseline and
    candidate have identical, wide, overlapping marginal distributions (so
    an unpaired test would see huge within-group variance and correctly
    fail to detect anything from the distributions alone), but every
    individual unit shifts by exactly the same small constant. A real
    paired test (resampling per-unit diffs) detects this easily, because the
    per-unit diffs have ~zero variance even though the raw values don't."""
    rng = np.random.default_rng(3)
    baseline = rng.normal(loc=100, scale=50, size=100)
    candidate = baseline + 5.0  # same units, constant shift, paired by index

    result = bootstrap_diff_ci(baseline.tolist(), candidate.tolist(), n_resamples=2000, random_state=3)

    assert result.significant is True
    assert result.observed_diff == pytest.approx(5.0, abs=0.01)


def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError, match="same length"):
        bootstrap_diff_ci([1.0, 2.0], [1.0])
