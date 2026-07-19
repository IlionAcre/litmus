import pytest
from scipy import stats as scipy_stats

from litmus.stats import mcnemar_test


def test_matches_hand_computed_reference_value():
    """Classic worked example: 10 pairs where baseline passed but candidate
    failed (b), 2 pairs where baseline failed but candidate passed (c), plus
    16 concordant pairs that shouldn't affect the result at all. Hand-computed
    continuity-corrected statistic: (|10-2|-1)^2 / (10+2) = 49/12 ~= 4.0833,
    independently recomputed here (not by calling into litmus.stats) and
    cross-checked against scipy's chi2 survival function directly."""
    baseline_passed = [True] * 10 + [False] * 2 + [True] * 8 + [False] * 8
    candidate_passed = [False] * 10 + [True] * 2 + [True] * 8 + [False] * 8

    result = mcnemar_test(baseline_passed, candidate_passed)

    expected_statistic = (abs(10 - 2) - 1) ** 2 / (10 + 2)
    expected_p_value = float(scipy_stats.chi2.sf(expected_statistic, df=1))

    assert result.b == 10
    assert result.c == 2
    assert result.statistic == pytest.approx(expected_statistic)
    assert result.p_value == pytest.approx(expected_p_value)
    assert result.significant is True  # p ~= 0.0433 < 0.05


def test_no_discordant_pairs_is_not_significant():
    baseline_passed = [True, True, False, False]
    candidate_passed = [True, True, False, False]

    result = mcnemar_test(baseline_passed, candidate_passed)

    assert result.b == 0
    assert result.c == 0
    assert result.p_value == 1.0
    assert result.significant is False


def test_small_discordant_counts_are_not_significant():
    # b=3, c=2: not enough discordant evidence to be significant.
    baseline_passed = [True] * 3 + [False] * 2 + [True] * 10
    candidate_passed = [False] * 3 + [True] * 2 + [True] * 10

    result = mcnemar_test(baseline_passed, candidate_passed)

    assert result.b == 3
    assert result.c == 2
    assert result.significant is False


def test_mismatched_lengths_raise_value_error():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_test([True, False], [True])
