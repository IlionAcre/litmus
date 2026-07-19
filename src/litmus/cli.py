import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from litmus.compare import compare_runs
from litmus.llm import litellm_call
from litmus.loader import load_test_cases
from litmus.runner import run_test_case
from litmus.schemas import RunResult, RunTarget, ScoreResult, TestCase
from litmus.scoring.registry import get_scorer
from litmus.storage import DEFAULT_RUNS_DIR, load_run, save_run

load_dotenv()

app = typer.Typer()


def _run_and_score(
    cases: list[TestCase], target: RunTarget
) -> tuple[list[RunResult], list[ScoreResult]]:
    results = []
    scores = []
    for case in cases:
        result = run_test_case(case, target, litellm_call)
        scorer = get_scorer(case.scorer)
        score_result = scorer.score(case, result)
        results.append(result)
        scores.append(score_result)
    return results, scores


@app.command()
def run(
    testset_dir: Path = typer.Argument(..., help="Directory of test case JSON files"),
    model: str = typer.Option(..., "--model", help="Model name passed to litellm"),
    prompt_version: str = typer.Option(
        "v1", "--prompt-version", help="Prompt version label for this run"
    ),
    runs_dir: Path = typer.Option(
        DEFAULT_RUNS_DIR, "--runs-dir", help="Directory to persist this run's results in"
    ),
) -> None:
    """Run a test set against a model, print each result, and persist the run."""
    cases = load_test_cases(testset_dir)
    target = RunTarget(prompt_version=prompt_version, model_name=model)
    results, scores = _run_and_score(cases, target)

    for result, score_result in zip(results, scores, strict=True):
        status = "PASS" if score_result.passed else "FAIL"
        typer.echo(
            f"[{status}] {result.test_case_id}: {result.raw_output!r} "
            f"({result.latency_ms:.1f}ms, ${result.cost_usd:.6f})"
        )

    persisted = save_run(target, results, scores, runs_dir=runs_dir)
    typer.echo(f"Saved run {persisted.run_id} to {runs_dir}")


@app.command()
def compare(
    baseline_run_id: str = typer.Argument(...),
    candidate_run_id: str = typer.Argument(...),
    runs_dir: Path = typer.Option(
        DEFAULT_RUNS_DIR, "--runs-dir", help="Directory persisted runs are read from"
    ),
) -> None:
    """Compare two already-persisted runs and print a statistically-grounded
    comparison report."""
    baseline = load_run(baseline_run_id, runs_dir=runs_dir)
    candidate = load_run(candidate_run_id, runs_dir=runs_dir)

    report = compare_runs(
        baseline.results, baseline.scores, candidate.results, candidate.scores
    )

    for metric in (report.pass_rate, report.latency_ms, report.cost_usd):
        flag = "REGRESSION" if metric.flagged else "ok"
        typer.echo(
            f"[{flag}] {metric.metric}: baseline={metric.baseline_mean:.4f} "
            f"candidate={metric.candidate_mean:.4f} delta={metric.delta:+.4f} "
            f"p={metric.p_value:.4g}"
        )

    if report.any_flagged:
        typer.echo("Result: REGRESSION DETECTED")
        raise typer.Exit(code=1)
    typer.echo("Result: no significant regression")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    runs_dir: Path = typer.Option(
        DEFAULT_RUNS_DIR, "--runs-dir", help="Directory persisted runs are read from"
    ),
) -> None:
    """Launch the local dashboard (FastAPI, reads persisted runs via DuckDB)."""
    import uvicorn

    os.environ["LITMUS_RUNS_DIR"] = str(runs_dir)
    uvicorn.run("litmus.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
