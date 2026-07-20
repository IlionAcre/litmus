import json
from itertools import count
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from litmus.cli import app

runner = CliRunner()


def _fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def _pin_deterministic_timing(monkeypatch):
    """Real wall-clock timing is flaky under system load and can spuriously
    trigger a "regression" on latency alone — pin it deterministically so
    tests only exercise the behavior they're actually meant to test."""
    counter = count(step=0.001)
    monkeypatch.setattr("time.perf_counter", lambda: next(counter))


def test_litmus_run_against_example_testset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response("positive"),
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)

    result = runner.invoke(
        app,
        [
            "run",
            "testsets/example",
            "--model",
            "gpt-4o-mini",
            "--prompt-version",
            "v1",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "case_001" in result.stdout
    assert "case_002" in result.stdout
    assert "positive" in result.stdout
    # case_001 expects "positive" and the mocked model always returns
    # "positive", so it should pass; case_002 expects "negative", so it
    # should fail — this exercises the exact-match scorer wiring end-to-end.
    assert "[PASS] case_001" in result.stdout
    assert "[FAIL] case_002" in result.stdout
    assert "Saved run" in result.stdout
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_litmus_run_requires_model_option():
    result = runner.invoke(app, ["run", "testsets/example"])

    assert result.exit_code != 0


def test_litmus_run_isolates_a_failing_case_instead_of_crashing_the_batch(monkeypatch, tmp_path):
    """A rate limit / auth / network error on one test case must not lose
    every already-computed result in the batch - it's recorded as [ERROR]
    and the run still gets persisted with everything else intact."""
    case_002_input = json.loads(Path("testsets/example/case_002.json").read_text())["input"]

    def fake_completion(model, messages):
        if messages[0]["content"] == case_002_input:
            raise RuntimeError("simulated rate limit error")
        return _fake_response("positive")

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)

    result = runner.invoke(
        app,
        [
            "run",
            "testsets/example",
            "--model",
            "gpt-4o-mini",
            "--prompt-version",
            "v1",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "[PASS] case_001" in result.stdout
    assert "[ERROR] case_002" in result.stdout
    assert "1 case(s) had errors" in result.stdout
    assert "simulated rate limit error" in result.stdout
    assert "Saved run" in result.stdout
    # the run is still persisted with both cases (one ok, one errored)
    assert len(list(tmp_path.glob("*.json"))) == 1
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert len(saved["results"]) == 2
    errored_result = next(r for r in saved["results"] if r["test_case_id"] == "case_002")
    assert errored_result["error"] is not None


def _write_synthetic_testset(tmp_path, n_positive: int = 10, n_negative: int = 10):
    """The 2-case example testset has no statistical power for a regression
    test (McNemar's needs enough discordant pairs to reach significance) — a
    larger synthetic set is needed here specifically."""
    testset_dir = tmp_path / "testset"
    testset_dir.mkdir()
    for i in range(n_positive):
        (testset_dir / f"pos_{i}.json").write_text(
            f'{{"id": "pos_{i}", "input": "great product #{i}", '
            f'"expected_output": "positive"}}'
        )
    for i in range(n_negative):
        (testset_dir / f"neg_{i}.json").write_text(
            f'{{"id": "neg_{i}", "input": "bad product #{i}", '
            f'"expected_output": "negative"}}'
        )
    return testset_dir


def _run(testset_dir, model, runs_dir, prompt_version="v1"):
    result = runner.invoke(
        app,
        [
            "run",
            str(testset_dir),
            "--model",
            model,
            "--prompt-version",
            prompt_version,
            "--runs-dir",
            str(runs_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    run_id = result.stdout.strip().splitlines()[-1].removeprefix(f"Saved run ").split(" to")[0]
    return run_id


def test_litmus_compare_detects_a_regression(monkeypatch, tmp_path):
    testset_dir = _write_synthetic_testset(tmp_path)
    runs_dir = tmp_path / "runs"

    # baseline model always answers correctly; candidate model always
    # answers "positive" regardless of the true label, so every
    # "negative"-expecting case flips from pass to fail -> a real regression.
    def fake_completion(model, messages):
        if model == "baseline-model":
            content = "positive" if "great" in messages[0]["content"] else "negative"
        else:
            content = "positive"
        return _fake_response(content)

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)
    _pin_deterministic_timing(monkeypatch)

    baseline_id = _run(testset_dir, "baseline-model", runs_dir)
    candidate_id = _run(testset_dir, "candidate-model", runs_dir)

    result = runner.invoke(
        app,
        ["compare", baseline_id, candidate_id, "--runs-dir", str(runs_dir)],
    )

    assert result.exit_code == 1
    assert "REGRESSION DETECTED" in result.stdout


def test_litmus_run_logs_structured_events(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response("positive"),
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)

    log_file = tmp_path / "run.jsonl"
    result = runner.invoke(
        app,
        [
            "--log-file",
            str(log_file),
            "run",
            "testsets/example",
            "--model",
            "gpt-4o-mini",
            "--prompt-version",
            "v1",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    events = [line["event"] for line in lines]
    assert "run_started" in events
    assert "run_completed" in events
    case_results = [line for line in lines if line["event"] == "case_result"]
    assert any(
        c["test_case_id"] == "case_001" and c["status"] == "PASS" for c in case_results
    )
    run_completed = next(line for line in lines if line["event"] == "run_completed")
    assert run_completed["total"] == 2


def test_litmus_run_logs_errored_case_result(monkeypatch, tmp_path):
    case_002_input = json.loads(Path("testsets/example/case_002.json").read_text())["input"]

    def fake_completion(model, messages):
        if messages[0]["content"] == case_002_input:
            raise RuntimeError("simulated rate limit error")
        return _fake_response("positive")

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)

    log_file = tmp_path / "run.jsonl"
    result = runner.invoke(
        app,
        [
            "--log-file",
            str(log_file),
            "run",
            "testsets/example",
            "--model",
            "gpt-4o-mini",
            "--prompt-version",
            "v1",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    case_results = [line for line in lines if line["event"] == "case_result"]
    errored = next(c for c in case_results if c["test_case_id"] == "case_002")
    assert errored["status"] == "ERROR"
    assert "simulated rate limit error" in errored["error"]
    run_completed = next(line for line in lines if line["event"] == "run_completed")
    assert run_completed["errored"] == 1


def test_litmus_compare_reports_no_regression_when_both_targets_agree(monkeypatch, tmp_path):
    testset_dir = _write_synthetic_testset(tmp_path)
    runs_dir = tmp_path / "runs"

    def fake_completion(model, messages):
        content = "positive" if "great" in messages[0]["content"] else "negative"
        return _fake_response(content)

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)
    _pin_deterministic_timing(monkeypatch)

    baseline_id = _run(testset_dir, "baseline-model", runs_dir)
    candidate_id = _run(testset_dir, "candidate-model", runs_dir)

    result = runner.invoke(
        app,
        ["compare", baseline_id, candidate_id, "--runs-dir", str(runs_dir)],
    )

    assert result.exit_code == 0
    assert "no significant regression" in result.stdout
