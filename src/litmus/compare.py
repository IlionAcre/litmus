from dataclasses import dataclass, field

from litmus.schemas import RunResult, ScoreResult
from litmus.stats import bootstrap_diff_ci, mcnemar_test


@dataclass
class MetricComparison:
    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    flagged: bool
    # pass_rate (McNemar's) sets p_value; latency_ms/cost_usd (paired
    # bootstrap) set ci_low/ci_high instead. Exactly one of these pairs is
    # populated depending on which statistical test produced this metric.
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass
class ComparisonReport:
    pass_rate: MetricComparison
    latency_ms: MetricComparison
    cost_usd: MetricComparison
    common_case_count: int
    baseline_only_ids: list[str] = field(default_factory=list)
    candidate_only_ids: list[str] = field(default_factory=list)
    errored_ids: list[str] = field(default_factory=list)

    @property
    def any_flagged(self) -> bool:
        return (
            self.pass_rate.flagged
            or self.latency_ms.flagged
            or self.cost_usd.flagged
        )

    @property
    def has_mismatched_cases(self) -> bool:
        """True iff the comparison excluded any test case — either because
        it wasn't present in both runs, or because it errored in one of
        them. Callers (CLI/API) must surface this, not bury it — a narrowed
        comparison that looks like a full one undermines the whole point of
        a statistical regression tool."""
        return bool(
            self.baseline_only_ids or self.candidate_only_ids or self.errored_ids
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


@dataclass
class _Alignment:
    aligned: list[tuple[RunResult, ScoreResult, RunResult, ScoreResult]]
    baseline_only_ids: list[str]
    candidate_only_ids: list[str]
    errored_ids: list[str]


def _align(
    baseline_results: list[RunResult],
    baseline_scores: list[ScoreResult],
    candidate_results: list[RunResult],
    candidate_scores: list[ScoreResult],
) -> _Alignment:
    """Align baseline and candidate records by test_case_id.

    A test case is included in the comparison only if it's present in both
    runs AND neither side recorded an error for it (see RunResult.error /
    ScoreResult.error). Everything excluded is reported back explicitly
    (baseline_only_ids / candidate_only_ids / errored_ids) rather than
    silently dropped — a testset changing between runs, or a case erroring,
    is normal usage, not something to hide.
    """
    baseline_results_by_id = {r.test_case_id: r for r in baseline_results}
    baseline_scores_by_id = {s.test_case_id: s for s in baseline_scores}
    candidate_results_by_id = {r.test_case_id: r for r in candidate_results}
    candidate_scores_by_id = {s.test_case_id: s for s in candidate_scores}

    baseline_ids = set(baseline_results_by_id) & set(baseline_scores_by_id)
    candidate_ids = set(candidate_results_by_id) & set(candidate_scores_by_id)

    baseline_only_ids = sorted(baseline_ids - candidate_ids)
    candidate_only_ids = sorted(candidate_ids - baseline_ids)
    shared_ids = baseline_ids & candidate_ids

    errored_ids = sorted(
        tid
        for tid in shared_ids
        if baseline_results_by_id[tid].error
        or baseline_scores_by_id[tid].error
        or candidate_results_by_id[tid].error
        or candidate_scores_by_id[tid].error
    )

    common_ids = shared_ids - set(errored_ids)

    aligned = [
        (
            baseline_results_by_id[tid],
            baseline_scores_by_id[tid],
            candidate_results_by_id[tid],
            candidate_scores_by_id[tid],
        )
        for tid in sorted(common_ids)
    ]

    return _Alignment(
        aligned=aligned,
        baseline_only_ids=baseline_only_ids,
        candidate_only_ids=candidate_only_ids,
        errored_ids=errored_ids,
    )


def compare_runs(
    baseline_results: list[RunResult],
    baseline_scores: list[ScoreResult],
    candidate_results: list[RunResult],
    candidate_scores: list[ScoreResult],
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> ComparisonReport:
    """Given a baseline and a candidate run (results + scores), align them by
    test_case_id and compute a statistically-grounded comparison report.

    pass_rate uses McNemar's test (paired pass/fail). latency_ms/cost_usd use
    a paired bootstrap CI for the mean difference, not Mann-Whitney U — both
    metrics are paired (same test case, before/after), and a paired bootstrap
    respects that pairing instead of treating baseline/candidate as
    independent samples (see CLAUDE.md)."""
    alignment = _align(
        baseline_results, baseline_scores, candidate_results, candidate_scores
    )
    aligned = alignment.aligned
    if not aligned:
        raise ValueError(
            "no common test_case_ids between baseline and candidate runs "
            "(after excluding cases only present in one run and cases that errored)"
        )

    baseline_passed = [bs.passed for _, bs, _, _ in aligned]
    candidate_passed = [cs.passed for _, _, _, cs in aligned]
    baseline_latency = [br.latency_ms for br, _, _, _ in aligned]
    candidate_latency = [cr.latency_ms for _, _, cr, _ in aligned]
    baseline_cost = [br.cost_usd for br, _, _, _ in aligned]
    candidate_cost = [cr.cost_usd for _, _, cr, _ in aligned]

    mcnemar_result = mcnemar_test(baseline_passed, candidate_passed, alpha=alpha)
    latency_result = bootstrap_diff_ci(
        baseline_latency, candidate_latency, confidence=confidence
    )
    cost_result = bootstrap_diff_ci(
        baseline_cost, candidate_cost, confidence=confidence
    )

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
            delta=latency_result.observed_diff,
            ci_low=latency_result.ci_low,
            ci_high=latency_result.ci_high,
            # A regression means latency got worse (higher), not just "changed".
            flagged=latency_result.significant and latency_result.observed_diff > 0,
        ),
        cost_usd=MetricComparison(
            metric="cost_usd",
            baseline_mean=baseline_mean_cost,
            candidate_mean=candidate_mean_cost,
            delta=cost_result.observed_diff,
            ci_low=cost_result.ci_low,
            ci_high=cost_result.ci_high,
            # A regression means cost got worse (higher), not just "changed".
            flagged=cost_result.significant and cost_result.observed_diff > 0,
        ),
        common_case_count=len(aligned),
        baseline_only_ids=alignment.baseline_only_ids,
        candidate_only_ids=alignment.candidate_only_ids,
        errored_ids=alignment.errored_ids,
    )
