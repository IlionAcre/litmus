import importlib
import json
from itertools import count
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

import litmus.cli as cli_module
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
    assert "(exact_binomial)" in result.stdout or "(chi_square)" in result.stdout


def test_litmus_compare_min_case_count_option_changes_power_warning(monkeypatch, tmp_path):
    """The comparison thresholds must actually be reachable from the CLI,
    not just exist as compare_runs() parameters nobody can override."""
    testset_dir = _write_synthetic_testset(tmp_path, n_positive=1, n_negative=1)
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(
        "litellm.completion", lambda model, messages: _fake_response("positive")
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)
    _pin_deterministic_timing(monkeypatch)

    baseline_id = _run(testset_dir, "same-model", runs_dir)
    candidate_id = _run(testset_dir, "same-model", runs_dir)

    default_result = runner.invoke(
        app, ["compare", baseline_id, candidate_id, "--runs-dir", str(runs_dir)]
    )
    assert "NOTE: low statistical power" in default_result.stdout

    overridden_result = runner.invoke(
        app,
        [
            "compare",
            baseline_id,
            candidate_id,
            "--runs-dir",
            str(runs_dir),
            "--min-case-count",
            "1",
            "--min-discordant-pairs",
            "0",
        ],
    )
    assert "NOTE: low statistical power" not in overridden_result.stdout


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


def test_litmus_run_with_invalid_testset_logs_and_exits_cleanly(tmp_path):
    bad_testset = tmp_path / "bad_testset"
    bad_testset.mkdir()
    (bad_testset / "broken.json").write_text("{not valid json")

    log_file = tmp_path / "run.jsonl"
    result = runner.invoke(
        app,
        [
            "--log-file",
            str(log_file),
            "run",
            str(bad_testset),
            "--model",
            "gpt-4o-mini",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 1
    assert "Error:" in result.stdout
    assert "Traceback" not in result.stdout

    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    failed = [line for line in lines if line["event"] == "testset_load_failed"]
    assert len(failed) == 1
    assert failed[0]["level"] == "ERROR"


def test_litmus_compare_with_missing_run_logs_and_exits_cleanly(tmp_path):
    log_file = tmp_path / "compare.jsonl"
    result = runner.invoke(
        app,
        [
            "--log-file",
            str(log_file),
            "compare",
            "does-not-exist-1",
            "does-not-exist-2",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 1
    assert "Error:" in result.stdout
    assert "Traceback" not in result.stdout

    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    failed = [line for line in lines if line["event"] == "run_load_failed"]
    assert len(failed) == 1
    assert failed[0]["level"] == "ERROR"


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


def test_litmus_run_concurrent_execution_matches_sequential_results(monkeypatch, tmp_path):
    """ThreadPoolExecutor.map() must preserve input order and produce the
    same per-case results as strictly sequential execution - concurrency is
    a performance change only, never a behavior change. Does NOT assert on
    latency_ms: _pin_deterministic_timing's shared itertools.count() is
    thread-safe for uniqueness but not deterministic per-case once workers
    interleave (verified empirically during planning - a sequential run
    gives every case the exact same latency by construction, a concurrent
    run gives varying, non-reproducible values), so this only asserts on
    ordering/content/pass-fail parity, per the plan's explicit instruction."""
    testset_dir = _write_synthetic_testset(tmp_path, n_positive=8, n_negative=8)

    def fake_completion(model, messages):
        content = "positive" if "great" in messages[0]["content"] else "negative"
        return _fake_response(content)

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)

    sequential_runs_dir = tmp_path / "sequential"
    concurrent_runs_dir = tmp_path / "concurrent"

    sequential_result = runner.invoke(
        app,
        [
            "run", str(testset_dir), "--model", "m",
            "--runs-dir", str(sequential_runs_dir), "--max-workers", "1",
        ],
    )
    concurrent_result = runner.invoke(
        app,
        [
            "run", str(testset_dir), "--model", "m",
            "--runs-dir", str(concurrent_runs_dir), "--max-workers", "4",
        ],
    )

    assert sequential_result.exit_code == 0
    assert concurrent_result.exit_code == 0

    sequential_saved = json.loads(next(sequential_runs_dir.glob("*.json")).read_text())
    concurrent_saved = json.loads(next(concurrent_runs_dir.glob("*.json")).read_text())

    sequential_ids = [r["test_case_id"] for r in sequential_saved["results"]]
    concurrent_ids = [r["test_case_id"] for r in concurrent_saved["results"]]
    assert sequential_ids == concurrent_ids

    sequential_pass = {s["test_case_id"]: s["passed"] for s in sequential_saved["scores"]}
    concurrent_pass = {s["test_case_id"]: s["passed"] for s in concurrent_saved["scores"]}
    assert sequential_pass == concurrent_pass


def test_litmus_run_isolates_a_scoring_failure_keeping_the_real_llm_result(monkeypatch, tmp_path):
    """Distinct from the LLM-call-failure test: here the LLM call succeeds
    (a real RunResult with real output/latency/cost) and only scoring itself
    blows up - the persisted RunResult must stay the real one, not a zeroed
    sentinel, while only the ScoreResult becomes an error placeholder."""
    testset_dir = tmp_path / "testset"
    testset_dir.mkdir()
    # No expected_output -> ExactMatchScorer.score() raises ValueError.
    (testset_dir / "no_expected.json").write_text(
        '{"id": "no_expected", "input": "whatever"}'
    )

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response("a real llm answer"),
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0002)

    result = runner.invoke(
        app,
        ["run", str(testset_dir), "--model", "m", "--runs-dir", str(tmp_path / "runs")],
    )

    assert result.exit_code == 0
    assert "[ERROR] no_expected" in result.stdout

    saved = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    saved_result = next(r for r in saved["results"] if r["test_case_id"] == "no_expected")
    saved_score = next(s for s in saved["scores"] if s["test_case_id"] == "no_expected")

    assert saved_result["error"] is None
    assert saved_result["raw_output"] == "a real llm answer"
    assert saved_result["cost_usd"] == 0.0002
    assert saved_score["error"] is not None
    assert "expected_output" in saved_score["error"]
    assert saved_score["passed"] is False


def test_litmus_compare_prints_mismatch_warning_with_excluded_ids(monkeypatch, tmp_path):
    baseline_dir = tmp_path / "baseline_testset"
    candidate_dir = tmp_path / "candidate_testset"
    baseline_dir.mkdir()
    candidate_dir.mkdir()

    # shared: case_a, case_b. baseline-only: case_c. candidate-only: case_d.
    for name in ("case_a", "case_b", "case_c"):
        (baseline_dir / f"{name}.json").write_text(
            f'{{"id": "{name}", "input": "x", "expected_output": "positive"}}'
        )
    for name in ("case_a", "case_b", "case_d"):
        (candidate_dir / f"{name}.json").write_text(
            f'{{"id": "{name}", "input": "x", "expected_output": "positive"}}'
        )

    monkeypatch.setattr(
        "litellm.completion", lambda model, messages: _fake_response("positive")
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)

    runs_dir = tmp_path / "runs"
    baseline_id = _run(baseline_dir, "m", runs_dir)
    candidate_id = _run(candidate_dir, "m", runs_dir)

    result = runner.invoke(
        app,
        ["compare", baseline_id, candidate_id, "--runs-dir", str(runs_dir)],
    )

    assert "WARNING: comparing 2 common test case(s)." in result.stdout
    assert "1 only in baseline (excluded): case_c" in result.stdout
    assert "1 only in candidate (excluded): case_d" in result.stdout


def test_litmus_compare_picks_up_litmus_toml_override_without_a_cli_flag(monkeypatch, tmp_path):
    """Config-file precedence must actually work end-to-end: CLI flag >
    litmus.toml > hardcoded fallback. Complements
    test_litmus_compare_min_case_count_option_changes_power_warning (which
    covers the CLI-flag layer) by proving the litmus.toml layer.
    typer.Option's default is evaluated once at module-import time from
    CONFIG, so exercising a config-file override requires actually reloading
    cli.py against a fake project directory, not just monkeypatching
    cli.CONFIG after the fact (that wouldn't touch the already-baked-in
    Option defaults)."""
    testset_dir = _write_synthetic_testset(tmp_path, n_positive=1, n_negative=1)
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(
        "litellm.completion", lambda model, messages: _fake_response("positive")
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)
    _pin_deterministic_timing(monkeypatch)

    baseline_id = _run(testset_dir, "same-model", runs_dir)
    candidate_id = _run(testset_dir, "same-model", runs_dir)

    (tmp_path / "litmus.toml").write_text(
        "min_case_count = 1\nmin_discordant_pairs = 0\n"
    )
    original_cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp_path)
        importlib.reload(cli_module)
        result = CliRunner().invoke(
            cli_module.app,
            ["compare", baseline_id, candidate_id, "--runs-dir", str(runs_dir)],
        )
    finally:
        # Restore cli_module's CONFIG/app to reflect the real project
        # directory before this test ends, regardless of pass/fail -
        # module reloads aren't undone by monkeypatch's own teardown.
        monkeypatch.chdir(original_cwd)
        importlib.reload(cli_module)

    assert "NOTE: low statistical power" not in result.stdout
