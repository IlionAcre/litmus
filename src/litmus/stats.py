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
    """Paired bootstrap CI for the mean difference (candidate - baseline).

    baseline[i] and candidate[i] must correspond to the same unit (e.g. the
    same test_case_id, before/after) — this is paired continuous data, not
    two independent samples, so it resamples the per-unit differences
    directly rather than resampling baseline and candidate independently
    (which would silently ignore the pairing, same methodological error as
    using Mann-Whitney U here — see CLAUDE.md)."""
    if len(baseline) != len(candidate):
        raise ValueError(
            "baseline and candidate must be the same length (they are paired per unit)"
        )

    rng = np.random.default_rng(random_state)
    diffs = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    observed_diff = float(diffs.mean())

    n = len(diffs)
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    resample_means = diffs[resample_indices].mean(axis=1)

    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(
        resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    significant = not (ci_low <= 0 <= ci_high)

    return BootstrapResult(
        observed_diff=observed_diff,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        significant=bool(significant),
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
