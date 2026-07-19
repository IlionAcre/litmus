from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from litmus.api import app
from litmus.schemas import RunResult, RunTarget, ScoreResult
from litmus.storage import save_run


def _result(test_case_id: str, latency_ms: float = 100.0, cost_usd: float = 0.001) -> RunResult:
    return RunResult(
        test_case_id=test_case_id,
        raw_output="x",
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        timestamp=datetime.now(UTC),
    )


def _score(test_case_id: str, passed: bool = True) -> ScoreResult:
    return ScoreResult(
        test_case_id=test_case_id,
        passed=passed,
        score=1.0 if passed else 0.0,
        explanation="",
    )


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("LITMUS_RUNS_DIR", str(tmp_path))
    return TestClient(app)


def test_trends_empty_when_no_runs(client):
    resp = client.get("/trends")

    assert resp.status_code == 200
    assert resp.json() == []


def test_get_run_returns_404_when_missing(client):
    resp = client.get("/runs/nonexistent")

    assert resp.status_code == 404


def test_get_run_returns_persisted_run(client, tmp_path):
    target = RunTarget(prompt_version="v1", model_name="m1")
    save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path, run_id="run1")

    resp = client.get("/runs/run1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "run1"
    assert body["results"][0]["test_case_id"] == "c1"
    assert body["scores"][0]["passed"] is True


def test_trends_returns_points_after_runs_exist(client, tmp_path):
    target = RunTarget(prompt_version="v1", model_name="m1")
    save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path, run_id="run1")

    resp = client.get("/trends")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["run_id"] == "run1"
    assert body[0]["pass_rate"] == 1.0


def test_compare_latest_requires_at_least_two_runs(client, tmp_path):
    target = RunTarget(prompt_version="v1", model_name="m1")
    save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path, run_id="run1")

    resp = client.get("/compare/latest")

    assert resp.status_code == 404


def test_compare_latest_compares_two_most_recently_created_runs(client, tmp_path):
    target = RunTarget(prompt_version="v1", model_name="m1")
    # Deliberately lexicographically out of order vs. creation order, to
    # exercise load_runs' chronological (not filename) ordering.
    save_run(target, [_result("c1")], [_score("c1", True)], runs_dir=tmp_path, run_id="z_first")
    save_run(target, [_result("c1")], [_score("c1", False)], runs_dir=tmp_path, run_id="a_second")

    resp = client.get("/compare/latest")

    assert resp.status_code == 200
    body = resp.json()
    assert body["baseline_run_id"] == "z_first"
    assert body["candidate_run_id"] == "a_second"
    assert "pass_rate" in body
    assert "any_flagged" in body


def test_compare_two_named_runs(client, tmp_path):
    target = RunTarget(prompt_version="v1", model_name="m1")
    save_run(target, [_result("c1")], [_score("c1", True)], runs_dir=tmp_path, run_id="a")
    save_run(target, [_result("c1")], [_score("c1", True)], runs_dir=tmp_path, run_id="b")

    resp = client.get("/compare/a/b")

    assert resp.status_code == 200
    body = resp.json()
    assert body["baseline_run_id"] == "a"
    assert body["candidate_run_id"] == "b"
    assert body["any_flagged"] is False


def test_compare_named_runs_404s_on_missing_run(client, tmp_path):
    target = RunTarget(prompt_version="v1", model_name="m1")
    save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path, run_id="a")

    resp = client.get("/compare/a/nonexistent")

    assert resp.status_code == 404
