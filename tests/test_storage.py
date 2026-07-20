import json
import logging
from datetime import UTC, datetime

import pytest

from litmus.logging_config import LOGGER_NAME, configure_logging
from litmus.schemas import RunResult, RunTarget, ScoreResult
from litmus.storage import load_run, load_runs, save_run


def _result(test_case_id: str) -> RunResult:
    return RunResult(
        test_case_id=test_case_id,
        raw_output="out",
        latency_ms=1.0,
        cost_usd=0.0001,
        timestamp=datetime.now(UTC),
    )


def _score(test_case_id: str, passed: bool = True) -> ScoreResult:
    return ScoreResult(
        test_case_id=test_case_id, passed=passed, score=1.0 if passed else 0.0, explanation=""
    )


def test_save_run_writes_a_json_file(tmp_path):
    target = RunTarget(prompt_version="v1", model_name="gpt-4o-mini")

    persisted = save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path)

    expected_path = tmp_path / f"{persisted.run_id}.json"
    assert expected_path.exists()
    assert not (tmp_path / f"{persisted.run_id}.json.tmp").exists()


def test_load_run_round_trips(tmp_path):
    target = RunTarget(prompt_version="v2", model_name="gpt-4o-mini")

    persisted = save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path)
    loaded = load_run(persisted.run_id, runs_dir=tmp_path)

    assert loaded.run_id == persisted.run_id
    assert loaded.prompt_version == "v2"
    assert loaded.model_name == "gpt-4o-mini"
    assert len(loaded.results) == 1
    assert loaded.results[0].test_case_id == "c1"
    assert loaded.scores[0].passed is True


def test_load_run_raises_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_run("nonexistent", runs_dir=tmp_path)


def test_running_twice_produces_two_persisted_files(tmp_path):
    target = RunTarget(prompt_version="v1", model_name="gpt-4o-mini")

    first = save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path)
    second = save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path)

    assert first.run_id != second.run_id
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_load_runs_returns_empty_list_when_directory_absent(tmp_path):
    absent = tmp_path / "does_not_exist"

    assert load_runs(runs_dir=absent) == []


def test_load_runs_returns_all_persisted_runs(tmp_path):
    target = RunTarget(prompt_version="v1", model_name="gpt-4o-mini")
    save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path, run_id="run_a")
    save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path, run_id="run_b")

    runs = load_runs(runs_dir=tmp_path)

    assert {r.run_id for r in runs} == {"run_a", "run_b"}


def test_load_runs_orders_chronologically_not_by_filename(tmp_path):
    """run_ids are random UUID4 hex by default, with no relationship to
    creation time — filename/glob order would be effectively random.
    Deliberately give the earlier-created run a lexicographically *later*
    id, so this test would fail if load_runs ever regressed to sorting by
    filename instead of created_at."""
    target = RunTarget(prompt_version="v1", model_name="gpt-4o-mini")
    save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path, run_id="z_created_first")
    save_run(target, [_result("c1")], [_score("c1")], runs_dir=tmp_path, run_id="a_created_second")

    runs = load_runs(runs_dir=tmp_path)

    assert [r.run_id for r in runs] == ["z_created_first", "a_created_second"]


def test_load_runs_logs_and_reraises_on_a_corrupt_run_file(tmp_path):
    """A corrupt/unparseable run file must still fail load_runs (behavior
    unchanged) but the failure must be visible in the structured log, not
    only as an uncaught exception."""
    log_file = tmp_path / "storage.jsonl"
    configure_logging(log_file, "INFO")

    (tmp_path / "corrupt.json").write_text("{not valid json")

    with pytest.raises(Exception):  # noqa: B017 - pydantic/json error, exact type not the point
        load_runs(runs_dir=tmp_path)

    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    failed = [line for line in lines if line["event"] == "run_load_failed"]
    assert len(failed) == 1
    assert failed[0]["level"] == "ERROR"
    assert "corrupt.json" in failed[0]["path"]
