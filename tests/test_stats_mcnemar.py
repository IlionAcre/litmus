import pytest
from scipy import stats as scipy_stats

from litmus.stats import mcnemar_test


def test_small_discordant_count_uses_exact_binomial_method():
    """b=10, c=2 (12 discordant pairs, below exact_threshold=25): the
    implementation now calls scipy.stats.binomtest directly for this range,
    so this is a direct equality check against that call - not an
    independent cross-check (see test_large_discordant_count_uses_chi_square
    below for the test that still provides genuine independent agreement).
    Confirmed by direct execution: exact p=0.03857421875 vs the chi-square
    approximation's p~=0.04331 - different values, same conclusion, which is
    exactly why small samples get routed to the exact test instead."""
    baseline_passed = [True] * 10 + [False] * 2 + [True] * 8 + [False] * 8
    candidate_passed = [False] * 10 + [True] * 2 + [True] * 8 + [False] * 8

    result = mcnemar_test(baseline_passed, candidate_passed)

    assert result.b == 10
    assert result.c == 2
    assert result.method == "exact_binomial"
    assert result.p_value == pytest.approx(0.03857421875, abs=1e-6)
    assert result.significant is True


def test_large_discordant_count_uses_chi_square_and_agrees_with_exact_method():
    """b=20, c=8 (28 discordant pairs, at/above exact_threshold=25): stays on
    the chi-square path, which provides the genuine independent cross-check
    the test above no longer offers. Confirmed by direct execution: exact
    p=0.03570 vs chi-square p=0.03764 - different formulas, close values,
    same conclusion."""
    baseline_passed = [True] * 20 + [False] * 8 + [True] * 30 + [False] * 42
    candidate_passed = [False] * 20 + [True] * 8 + [True] * 30 + [False] * 42

    result = mcnemar_test(baseline_passed, candidate_passed)

    exact = scipy_stats.binomtest(8, 20 + 8, 0.5, alternative="two-sided")

    assert result.b == 20
    assert result.c == 8
    assert result.method == "chi_square"
    assert result.significant is True
    assert exact.pvalue < 0.05  # independent method agrees: significant
    assert result.p_value == pytest.approx(exact.pvalue, abs=0.01)  # close, not identical


def test_no_discordant_pairs_is_not_significant():
    baseline_passed = [True, True, False, False]
    candidate_passed = [True, True, False, False]

    result = mcnemar_test(baseline_passed, candidate_passed)

    assert result.b == 0
    assert result.c == 0
    assert result.p_value == 1.0
    assert result.significant is False
    assert result.method == "no_discordant_pairs"


def test_small_discordant_counts_are_not_significant():
    # b=3, c=2 (5 discordant pairs, below exact_threshold=25): exact-binomial
    # path. Confirmed by direct execution: both the exact and chi-square
    # methods give p=1.0 here - not enough discordant evidence either way.
    baseline_passed = [True] * 3 + [False] * 2 + [True] * 10
    candidate_passed = [False] * 3 + [True] * 2 + [True] * 10

    result = mcnemar_test(baseline_passed, candidate_passed)

    assert result.b == 3
    assert result.c == 2
    assert result.method == "exact_binomial"
    assert result.significant is False


def test_exact_threshold_boundary():
    """b+c one below the default exact_threshold=25 takes the exact path;
    b+c at the threshold takes the chi-square path."""
    below = mcnemar_test([True] * 12 + [False] * 12, [False] * 12 + [True] * 12)
    assert below.b + below.c == 24
    assert below.method == "exact_binomial"

    at_threshold = mcnemar_test(
        [True] * 12 + [False] * 13, [False] * 12 + [True] * 13
    )
    assert at_threshold.b + at_threshold.c == 25
    assert at_threshold.method == "chi_square"


def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_test([True, False], [True])
