from datetime import UTC, datetime

import numpy as np
import pytest

from litmus.compare import compare_runs
from litmus.schemas import RunResult, ScoreResult


def _results(ids: list[str], latency: float, cost: float) -> list[RunResult]:
    return [
        RunResult(
            test_case_id=tid,
            raw_output="output",
            latency_ms=latency,
            cost_usd=cost,
            timestamp=datetime.now(UTC),
        )
        for tid in ids
    ]


def _results_with_variance(
    ids: list[str], mean_latency: float, mean_cost: float, rng: np.random.Generator
) -> list[RunResult]:
    """Like _results, but with realistic per-case jitter — needed so
    Mann-Whitney has genuine within-group variance to work with. Constant
    values across every case have zero variance, so Mann-Whitney correctly
    (not spuriously) treats *any* consistent shift as maximally significant,
    which makes a "pure noise" test meaningless without real variance."""
    return [
        RunResult(
            test_case_id=tid,
            raw_output="output",
            latency_ms=max(1.0, float(rng.normal(mean_latency, mean_latency * 0.1))),
            cost_usd=max(0.0001, float(rng.normal(mean_cost, mean_cost * 0.1))),
            timestamp=datetime.now(UTC),
        )
        for tid in ids
    ]


def _scores(ids: list[str], passed: list[bool]) -> list[ScoreResult]:
    return [
        ScoreResult(test_case_id=tid, passed=p, score=1.0 if p else 0.0, explanation="")
        for tid, p in zip(ids, passed, strict=True)
    ]


def test_detects_a_real_pass_rate_regression():
    ids = [f"c{i}" for i in range(20)]
    baseline_passed = [True] * 18 + [False] * 2
    candidate_passed = [False] * 10 + [True] * 10  # much worse

    baseline_results = _results(ids, latency=100.0, cost=0.001)
    candidate_results = _results(ids, latency=100.0, cost=0.001)

    report = compare_runs(
        baseline_results,
        _scores(ids, baseline_passed),
        candidate_results,
        _scores(ids, candidate_passed),
    )

    assert report.pass_rate.flagged is True
    assert report.pass_rate.delta < 0
    assert report.any_flagged is True


def test_does_not_flag_noise_when_nothing_meaningfully_changed():
    ids = [f"c{i}" for i in range(50)]
    passed = [True] * 45 + [False] * 5
    rng = np.random.default_rng(123)

    # Same underlying distribution for both runs -> any difference observed
    # is just sampling noise, not a real regression.
    baseline_results = _results_with_variance(ids, mean_latency=100.0, mean_cost=0.001, rng=rng)
    candidate_results = _results_with_variance(ids, mean_latency=100.0, mean_cost=0.001, rng=rng)

    report = compare_runs(
        baseline_results,
        _scores(ids, passed),
        candidate_results,
        _scores(ids, passed),
    )

    assert report.pass_rate.flagged is False
    assert report.latency_ms.flagged is False
    assert report.cost_usd.flagged is False
    assert report.any_flagged is False


def test_does_not_flag_an_improvement():
    ids = [f"c{i}" for i in range(20)]
    baseline_passed = [False] * 10 + [True] * 10
    candidate_passed = [True] * 18 + [False] * 2  # candidate is much better

    baseline_results = _results(ids, latency=100.0, cost=0.001)
    candidate_results = _results(ids, latency=100.0, cost=0.001)

    report = compare_runs(
        baseline_results,
        _scores(ids, baseline_passed),
        candidate_results,
        _scores(ids, candidate_passed),
    )

    assert report.pass_rate.flagged is False


def test_detects_a_real_latency_regression():
    ids = [f"c{i}" for i in range(30)]
    passed = [True] * 30
    rng = np.random.default_rng(99)

    baseline_results = _results_with_variance(ids, mean_latency=100.0, mean_cost=0.001, rng=rng)
    candidate_results = _results_with_variance(ids, mean_latency=400.0, mean_cost=0.001, rng=rng)

    report = compare_runs(
        baseline_results,
        _scores(ids, passed),
        candidate_results,
        _scores(ids, passed),
    )

    assert report.latency_ms.flagged is True
    assert report.latency_ms.delta > 0


def test_raises_when_no_common_test_case_ids():
    baseline_results = _results(["a"], latency=100.0, cost=0.001)
    candidate_results = _results(["b"], latency=100.0, cost=0.001)

    with pytest.raises(ValueError, match="no common test_case_ids"):
        compare_runs(
            baseline_results,
            _scores(["a"], [True]),
            candidate_results,
            _scores(["b"], [True]),
        )


def test_only_compares_common_test_case_ids():
    baseline_results = _results(["a", "b", "extra_baseline_only"], latency=100.0, cost=0.001)
    candidate_results = _results(["a", "b", "extra_candidate_only"], latency=100.0, cost=0.001)

    report = compare_runs(
        baseline_results,
        _scores(["a", "b", "extra_baseline_only"], [True, True, True]),
        candidate_results,
        _scores(["a", "b", "extra_candidate_only"], [True, True, False]),
    )

    # only "a" and "b" are common; both passed in both runs -> no regression
    assert report.pass_rate.flagged is False
