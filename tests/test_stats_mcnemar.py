import pytest
from scipy import stats as scipy_stats

from litmus.stats import mcnemar_test


def test_cross_checked_against_exact_binomial_mcnemar():
    """Independent cross-check using a genuinely different method, not the
    continuity-corrected chi-square formula under test: the *exact* McNemar
    test via a two-sided binomial test on the discordant pairs against
    p=0.5 (scipy.stats.binomtest) — a different, independently-derived
    procedure for the same null hypothesis (b and c pairs are equally
    likely), not a re-expression of litmus.stats' own formula.

    For b=10, c=2 (12 discordant pairs): scipy.stats.binomtest(2, 12, 0.5,
    alternative="two-sided").pvalue ~= 0.03857 (verified by direct
    execution), and this implementation's continuity-corrected chi-square
    gives p ~= 0.04331. Different formulas, same conclusion (both < 0.05) -
    that agreement, not shared arithmetic, is the actual independent check.
    """
    baseline_passed = [True] * 10 + [False] * 2 + [True] * 8 + [False] * 8
    candidate_passed = [False] * 10 + [True] * 2 + [True] * 8 + [False] * 8

    result = mcnemar_test(baseline_passed, candidate_passed)

    exact = scipy_stats.binomtest(2, 10 + 2, 0.5, alternative="two-sided")

    assert result.b == 10
    assert result.c == 2
    assert result.significant is True  # p ~= 0.0433 < 0.05
    assert exact.pvalue < 0.05  # the exact test agrees: significant
    assert exact.pvalue == pytest.approx(0.03857421875, abs=1e-6)


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
