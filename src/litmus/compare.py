from dataclasses import dataclass

from litmus.schemas import RunResult, ScoreResult
from litmus.stats import mann_whitney_test, mcnemar_test


@dataclass
class MetricComparison:
    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    p_value: float
    flagged: bool


@dataclass
class ComparisonReport:
    pass_rate: MetricComparison
    latency_ms: MetricComparison
    cost_usd: MetricComparison

    @property
    def any_flagged(self) -> bool:
        return (
            self.pass_rate.flagged
            or self.latency_ms.flagged
            or self.cost_usd.flagged
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _align(
    baseline_results: list[RunResult],
    baseline_scores: list[ScoreResult],
    candidate_results: list[RunResult],
    candidate_scores: list[ScoreResult],
) -> list[tuple[RunResult, ScoreResult, RunResult, ScoreResult]]:
    """Align baseline and candidate records by test_case_id, dropping any
    test case not present in all four inputs."""
    baseline_results_by_id = {r.test_case_id: r for r in baseline_results}
    baseline_scores_by_id = {s.test_case_id: s for s in baseline_scores}
    candidate_results_by_id = {r.test_case_id: r for r in candidate_results}
    candidate_scores_by_id = {s.test_case_id: s for s in candidate_scores}

    common_ids = (
        set(baseline_results_by_id)
        & set(baseline_scores_by_id)
        & set(candidate_results_by_id)
        & set(candidate_scores_by_id)
    )

    return [
        (
            baseline_results_by_id[tid],
            baseline_scores_by_id[tid],
            candidate_results_by_id[tid],
            candidate_scores_by_id[tid],
        )
        for tid in sorted(common_ids)
    ]


def compare_runs(
    baseline_results: list[RunResult],
    baseline_scores: list[ScoreResult],
    candidate_results: list[RunResult],
    candidate_scores: list[ScoreResult],
    alpha: float = 0.05,
) -> ComparisonReport:
    """Given a baseline and a candidate run (results + scores), align them by
    test_case_id and compute a statistically-grounded comparison report."""
    aligned = _align(
        baseline_results, baseline_scores, candidate_results, candidate_scores
    )
    if not aligned:
        raise ValueError(
            "no common test_case_ids between baseline and candidate runs"
        )

    baseline_passed = [bs.passed for _, bs, _, _ in aligned]
    candidate_passed = [cs.passed for _, _, _, cs in aligned]
    baseline_latency = [br.latency_ms for br, _, _, _ in aligned]
    candidate_latency = [cr.latency_ms for _, _, cr, _ in aligned]
    baseline_cost = [br.cost_usd for br, _, _, _ in aligned]
    candidate_cost = [cr.cost_usd for _, _, cr, _ in aligned]

    mcnemar_result = mcnemar_test(baseline_passed, candidate_passed, alpha=alpha)
    latency_result = mann_whitney_test(baseline_latency, candidate_latency, alpha=alpha)
    cost_result = mann_whitney_test(baseline_cost, candidate_cost, alpha=alpha)

    baseline_pass_rate = _mean([float(p) for p in baseline_passed])
    candidate_pass_rate = _mean([float(p) for p in candidate_passed])
    baseline_mean_latency = _mean(baseline_latency)
    candidate_mean_latency = _mean(candidate_latency)
    baseline_mean_cost = _mean(baseline_cost)
    candidate_mean_cost = _mean(candidate_cost)

    return ComparisonReport(
        pass_rate=MetricComparison(
            metric="pass_rate",
            baseline_mean=baseline_pass_rate,
            candidate_mean=candidate_pass_rate,
            delta=candidate_pass_rate - baseline_pass_rate,
            p_value=mcnemar_result.p_value,
            # A regression means pass rate got worse, not just "changed".
            flagged=mcnemar_result.significant
            and candidate_pass_rate < baseline_pass_rate,
        ),
        latency_ms=MetricComparison(
            metric="latency_ms",
            baseline_mean=baseline_mean_latency,
            candidate_mean=candidate_mean_latency,
            delta=candidate_mean_latency - baseline_mean_latency,
            p_value=latency_result.p_value,
            # A regression means latency got worse (higher), not just "changed".
            flagged=latency_result.significant
            and candidate_mean_latency > baseline_mean_latency,
        ),
        cost_usd=MetricComparison(
            metric="cost_usd",
            baseline_mean=baseline_mean_cost,
            candidate_mean=candidate_mean_cost,
            delta=candidate_mean_cost - baseline_mean_cost,
            p_value=cost_result.p_value,
            # A regression means cost got worse (higher), not just "changed".
            flagged=cost_result.significant and candidate_mean_cost > baseline_mean_cost,
        ),
    )
