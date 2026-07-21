import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv

from litmus.compare import compare_runs
from litmus.llm import litellm_call
from litmus.loader import TestCaseLoadError, load_test_cases
from litmus.logging_config import LOGGER_NAME, configure_logging
from litmus.runner import run_test_case
from litmus.schemas import RunResult, RunTarget, ScoreResult, TestCase
from litmus.scoring.registry import get_scorer
from litmus.storage import DEFAULT_RUNS_DIR, load_run, save_run

load_dotenv()

app = typer.Typer()
logger = logging.getLogger(LOGGER_NAME)


@app.callback()
def main(
    log_file: Path = typer.Option(
        Path("logs/litmus.jsonl"),
        "--log-file",
        envvar="LITMUS_LOG_FILE",
        help="Structured JSON-lines log file path",
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", envvar="LITMUS_LOG_LEVEL", help="Logging level"
    ),
) -> None:
    """Litmus: an LLM evaluation and regression-testing tool."""
    configure_logging(log_file, log_level)


def _run_and_score(
    cases: list[TestCase], target: RunTarget
) -> tuple[list[RunResult], list[ScoreResult]]:
    """Run and score every case, isolating failures per-case: a single
    case's LLM call or scoring blowing up (rate limit, bad model name, auth
    failure, an unparseable judge response, ...) must not lose every
    already-computed result in the batch. A failed case is recorded with
    `error` set (see RunResult/ScoreResult) instead of crashing the run."""
    results = []
    scores = []
    for case in cases:
        try:
            result = run_test_case(case, target, litellm_call)
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            error_msg = f"{type(e).__name__}: {e}"
            results.append(
                RunResult(
                    test_case_id=case.id,
                    raw_output="",
                    latency_ms=0.0,
                    cost_usd=0.0,
                    timestamp=datetime.now(UTC),
                    error=error_msg,
                )
            )
            scores.append(
                ScoreResult(
                    test_case_id=case.id,
                    passed=False,
                    score=0.0,
                    explanation="not scored: the run itself failed",
                    error=error_msg,
                )
            )
            continue

        try:
            scorer = get_scorer(case.scorer)
            score_result = scorer.score(case, result)
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            error_msg = f"{type(e).__name__}: {e}"
            score_result = ScoreResult(
                test_case_id=case.id,
                passed=False,
                score=0.0,
                explanation="scoring failed",
                error=error_msg,
            )

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
    """Run a test set against a model, print each result, and persist the
    run. A case that errors (LLM call or scoring failure) doesn't abort the
    batch — it's recorded as [ERROR] and the run is still saved with
    whatever cases did succeed."""
    try:
        cases = load_test_cases(testset_dir)
    except TestCaseLoadError as e:
        logger.error(
            "failed to load test cases",
            extra={
                "event": "testset_load_failed",
                "testset_dir": testset_dir,
                "error": str(e),
            },
        )
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e

    target = RunTarget(prompt_version=prompt_version, model_name=model)
    logger.info(
        "run started",
        extra={
            "event": "run_started",
            "testset_dir": testset_dir,
            "model": model,
            "prompt_version": prompt_version,
            "case_count": len(cases),
        },
    )
    results, scores = _run_and_score(cases, target)

    for result, score_result in zip(results, scores, strict=True):
        if result.error or score_result.error:
            status = "ERROR"
        else:
            status = "PASS" if score_result.passed else "FAIL"
        typer.echo(
            f"[{status}] {result.test_case_id}: {result.raw_output!r} "
            f"({result.latency_ms:.1f}ms, ${result.cost_usd:.6f})"
        )
        logger.log(
            logging.ERROR if status == "ERROR" else logging.INFO,
            f"case {status}: {result.test_case_id}",
            extra={
                "event": "case_result",
                "test_case_id": result.test_case_id,
                "status": status,
                "latency_ms": result.latency_ms,
                "cost_usd": result.cost_usd,
                "error": result.error or score_result.error,
            },
        )

    errored = [
        (r, s)
        for r, s in zip(results, scores, strict=True)
        if r.error or s.error
    ]
    if errored:
        typer.echo("")
        typer.echo(f"{len(errored)} case(s) had errors:")
        for result, score_result in errored:
            typer.echo(f"  [ERROR] {result.test_case_id}: {result.error or score_result.error}")

    persisted = save_run(target, results, scores, runs_dir=runs_dir)
    typer.echo(f"Saved run {persisted.run_id} to {runs_dir}")

    passed_count = sum(
        1
        for r, s in zip(results, scores, strict=True)
        if not (r.error or s.error) and s.passed
    )
    logger.info(
        "run completed",
        extra={
            "event": "run_completed",
            "run_id": persisted.run_id,
            "total": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count - len(errored),
            "errored": len(errored),
        },
    )


@app.command()
def compare(
    baseline_run_id: str = typer.Argument(...),
    candidate_run_id: str = typer.Argument(...),
    runs_dir: Path = typer.Option(
        DEFAULT_RUNS_DIR, "--runs-dir", help="Directory persisted runs are read from"
    ),
    alpha: float = typer.Option(
        0.05, "--alpha", help="Significance level for McNemar's test (pass_rate)"
    ),
    confidence: float = typer.Option(
        0.95, "--confidence", help="Confidence level for the bootstrap CIs"
    ),
    min_case_count: int = typer.Option(
        10,
        "--min-case-count",
        help="power_warning threshold: minimum common test cases expected",
    ),
    min_discordant_pairs: int = typer.Option(
        10,
        "--min-discordant-pairs",
        help="power_warning threshold: minimum McNemar discordant pairs expected",
    ),
    exact_threshold: int = typer.Option(
        25,
        "--exact-threshold",
        help="Below this many discordant pairs, use the exact binomial test "
        "instead of the chi-square approximation for pass_rate",
    ),
) -> None:
    """Compare two already-persisted runs and print a statistically-grounded
    comparison report."""
    try:
        baseline = load_run(baseline_run_id, runs_dir=runs_dir)
        candidate = load_run(candidate_run_id, runs_dir=runs_dir)
    except FileNotFoundError as e:
        logger.error(
            "failed to load run for comparison",
            extra={"event": "run_load_failed", "error": str(e)},
        )
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e

    report = compare_runs(
        baseline.results,
        baseline.scores,
        candidate.results,
        candidate.scores,
        alpha=alpha,
        confidence=confidence,
        min_case_count=min_case_count,
        min_discordant_pairs=min_discordant_pairs,
        exact_threshold=exact_threshold,
    )
    logger.info(
        "comparison performed",
        extra={
            "event": "comparison_performed",
            "baseline_run_id": baseline_run_id,
            "candidate_run_id": candidate_run_id,
            "common_case_count": report.common_case_count,
            "any_flagged": report.any_flagged,
            "pass_rate_delta": report.pass_rate.delta,
            "pass_rate_flagged": report.pass_rate.flagged,
            "latency_ms_delta": report.latency_ms.delta,
            "latency_ms_flagged": report.latency_ms.flagged,
            "cost_usd_delta": report.cost_usd.delta,
            "cost_usd_flagged": report.cost_usd.flagged,
            "mean_score_delta": report.mean_score.delta,
            "mean_score_flagged": report.mean_score.flagged,
            "power_warning": report.power_warning,
        },
    )

    if report.has_mismatched_cases:
        logger.warning(
            "comparison has mismatched/errored cases",
            extra={
                "event": "comparison_mismatch",
                "baseline_only_count": len(report.baseline_only_ids),
                "candidate_only_count": len(report.candidate_only_ids),
                "errored_count": len(report.errored_ids),
            },
        )
        typer.echo(f"WARNING: comparing {report.common_case_count} common test case(s).")
        if report.baseline_only_ids:
            typer.echo(
                f"  {len(report.baseline_only_ids)} only in baseline "
                f"(excluded): {', '.join(report.baseline_only_ids)}"
            )
        if report.candidate_only_ids:
            typer.echo(
                f"  {len(report.candidate_only_ids)} only in candidate "
                f"(excluded): {', '.join(report.candidate_only_ids)}"
            )
        if report.errored_ids:
            typer.echo(
                f"  {len(report.errored_ids)} excluded due to errors: "
                f"{', '.join(report.errored_ids)}"
            )
        typer.echo("")

    if report.power_warning:
        typer.echo(f"NOTE: low statistical power - {report.power_warning}.")
        typer.echo("")

    for metric in (report.pass_rate, report.latency_ms, report.cost_usd, report.mean_score):
        flag = "REGRESSION" if metric.flagged else "ok"
        if metric.p_value is not None:
            method_suffix = f" ({metric.method})" if metric.method else ""
            typer.echo(
                f"[{flag}] {metric.metric}: baseline={metric.baseline_mean:.4f} "
                f"candidate={metric.candidate_mean:.4f} delta={metric.delta:+.4f} "
                f"p={metric.p_value:.4g}{method_suffix}"
            )
        else:
            typer.echo(
                f"[{flag}] {metric.metric}: baseline={metric.baseline_mean:.4f} "
                f"candidate={metric.candidate_mean:.4f} delta={metric.delta:+.4f} "
                f"95% CI=[{metric.ci_low:.4g}, {metric.ci_high:.4g}]"
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
