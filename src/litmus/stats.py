from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class BootstrapResult:
    observed_diff: float
    ci_low: float
    ci_high: float
    significant: bool  # True iff the CI excludes zero


@dataclass
class MannWhitneyResult:
    statistic: float
    p_value: float
    significant: bool


@dataclass
class McNemarResult:
    b: int  # baseline passed, candidate failed
    c: int  # baseline failed, candidate passed
    statistic: float
    p_value: float
    significant: bool


def bootstrap_diff_ci(
    baseline: list[float],
    candidate: list[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> BootstrapResult:
    """Paired-independent bootstrap CI for the difference in means
    (candidate - baseline). Vectorized for speed at realistic n_resamples."""
    rng = np.random.default_rng(random_state)
    baseline_arr = np.asarray(baseline, dtype=float)
    candidate_arr = np.asarray(candidate, dtype=float)

    observed_diff = float(candidate_arr.mean() - baseline_arr.mean())

    b_samples = rng.choice(
        baseline_arr, size=(n_resamples, len(baseline_arr)), replace=True
    )
    c_samples = rng.choice(
        candidate_arr, size=(n_resamples, len(candidate_arr)), replace=True
    )
    diffs = c_samples.mean(axis=1) - b_samples.mean(axis=1)

    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    significant = not (ci_low <= 0 <= ci_high)

    return BootstrapResult(
        observed_diff=observed_diff,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        significant=bool(significant),
    )


def mann_whitney_test(
    baseline: list[float],
    candidate: list[float],
    alpha: float = 0.05,
) -> MannWhitneyResult:
    """Mann-Whitney U test for whether candidate's distribution differs from
    baseline's (two-sided). Used for latency/cost deltas."""
    statistic, p_value = scipy_stats.mannwhitneyu(
        candidate, baseline, alternative="two-sided"
    )
    return MannWhitneyResult(
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < alpha),
    )


def mcnemar_test(
    baseline_passed: list[bool],
    candidate_passed: list[bool],
    alpha: float = 0.05,
) -> McNemarResult:
    """Manually-implemented McNemar's test (with continuity correction) for
    paired pass/fail-rate comparisons. Hand-rolled deliberately — see
    CLAUDE.md: no `statsmodels` dependency for this one function."""
    if len(baseline_passed) != len(candidate_passed):
        raise ValueError(
            "baseline_passed and candidate_passed must be the same length "
            "(they are paired per test case)"
        )

    b = sum(
        1
        for base, cand in zip(baseline_passed, candidate_passed, strict=True)
        if base and not cand
    )
    c = sum(
        1
        for base, cand in zip(baseline_passed, candidate_passed, strict=True)
        if not base and cand
    )

    if b + c == 0:
        # No discordant pairs at all: no evidence of any difference.
        return McNemarResult(b=b, c=c, statistic=0.0, p_value=1.0, significant=False)

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(scipy_stats.chi2.sf(statistic, df=1))

    return McNemarResult(
        b=b,
        c=c,
        statistic=float(statistic),
        p_value=p_value,
        significant=bool(p_value < alpha),
    )
