from datetime import UTC, datetime

import pytest

from litmus.schemas import RunResult, RunTarget, ScoreResult
from litmus.storage import save_run
from litmus.trends import query_trends


def _result(test_case_id: str, latency_ms: float, cost_usd: float) -> RunResult:
    return RunResult(
        test_case_id=test_case_id,
        raw_output="x",
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        timestamp=datetime.now(UTC),
    )


def _score(test_case_id: str, passed: bool) -> ScoreResult:
    return ScoreResult(
        test_case_id=test_case_id,
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation="",
    )


def test_returns_empty_list_when_no_runs_directory(tmp_path):
    assert query_trends(runs_dir=tmp_path / "absent") == []


def test_returns_empty_list_when_runs_directory_is_empty(tmp_path):
    assert query_trends(runs_dir=tmp_path) == []


def test_computes_correct_aggregates_for_one_run(tmp_path):
    target = RunTarget(prompt_version="v1", model_name="gpt-4o-mini")
    save_run(
        target,
        [_result("c1", 100.0, 0.001), _result("c2", 200.0, 0.002)],
        [_score("c1", True), _score("c2", False)],
        runs_dir=tmp_path,
        run_id="run1",
    )

    points = query_trends(runs_dir=tmp_path)

    assert len(points) == 1
    point = points[0]
    assert point.run_id == "run1"
    assert point.prompt_version == "v1"
    assert point.model_name == "gpt-4o-mini"
    assert point.pass_rate == pytest.approx(0.5)
    assert point.mean_latency_ms == pytest.approx(150.0)
    assert point.mean_cost_usd == pytest.approx(0.0015)


def test_handles_a_run_with_zero_test_cases_without_crashing(tmp_path):
    """A run with empty results/scores makes DuckDB infer a generic JSON
    type for that file's empty array instead of a typed struct list, which
    breaks avg() unless the schema is declared explicitly. Regression test
    for that bug, found by simulating the Phase 8 CI baseline-lookup path."""
    target = RunTarget(prompt_version="v1", model_name="m1")
    save_run(target, [], [], runs_dir=tmp_path, run_id="empty_run")

    points = query_trends(runs_dir=tmp_path)

    assert len(points) == 1
    assert points[0].run_id == "empty_run"
    assert points[0].pass_rate is None
    assert points[0].mean_latency_ms is None
    assert points[0].mean_cost_usd is None


def test_handles_a_mix_of_populated_and_empty_runs(tmp_path):
    target = RunTarget(prompt_version="v1", model_name="m1")
    save_run(
        target, [_result("c1", 100.0, 0.001)], [_score("c1", True)],
        runs_dir=tmp_path, run_id="populated_run",
    )
    save_run(target, [], [], runs_dir=tmp_path, run_id="empty_run")

    points = query_trends(runs_dir=tmp_path)
    by_id = {p.run_id: p for p in points}

    assert by_id["populated_run"].pass_rate == pytest.approx(1.0)
    assert by_id["empty_run"].pass_rate is None


def test_orders_multiple_runs_chronologically(tmp_path):
    target = RunTarget(prompt_version="v1", model_name="m1")

    save_run(
        target, [_result("c1", 100.0, 0.001)], [_score("c1", True)],
        runs_dir=tmp_path, run_id="run_a",
    )
    save_run(
        target, [_result("c1", 300.0, 0.003)], [_score("c1", True)],
        runs_dir=tmp_path, run_id="run_b",
    )

    points = query_trends(runs_dir=tmp_path)

    assert [p.run_id for p in points] == ["run_a", "run_b"]
    assert points[0].created_at <= points[1].created_at
    assert points[1].mean_latency_ms == pytest.approx(300.0)
