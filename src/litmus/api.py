import logging
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException

from litmus.compare import ComparisonReport, compare_runs
from litmus.logging_config import LOGGER_NAME
from litmus.storage import DEFAULT_RUNS_DIR, load_run, load_runs
from litmus.trends import query_trends

app = FastAPI(title="Litmus Dashboard")
logger = logging.getLogger(LOGGER_NAME)


def _runs_dir() -> Path:
    """Resolved at request time (not import time) via LITMUS_RUNS_DIR so
    tests can point the API at an isolated directory."""
    return Path(os.environ.get("LITMUS_RUNS_DIR", str(DEFAULT_RUNS_DIR)))


def _report_dict(report: ComparisonReport) -> dict:
    return {
        "pass_rate": asdict(report.pass_rate),
        "latency_ms": asdict(report.latency_ms),
        "cost_usd": asdict(report.cost_usd),
        "any_flagged": report.any_flagged,
        "common_case_count": report.common_case_count,
        "baseline_only_ids": report.baseline_only_ids,
        "candidate_only_ids": report.candidate_only_ids,
        "errored_ids": report.errored_ids,
    }


@app.get("/trends")
def get_trends() -> list[dict]:
    """Historical trend points (pass rate, mean latency, mean cost) across
    every persisted run, ordered chronologically."""
    logger.info("GET /trends", extra={"event": "api_request", "route": "/trends"})
    points = query_trends(runs_dir=_runs_dir())
    return [
        {
            "run_id": p.run_id,
            "prompt_version": p.prompt_version,
            "model_name": p.model_name,
            "created_at": p.created_at.isoformat(),
            "pass_rate": p.pass_rate,
            "mean_latency_ms": p.mean_latency_ms,
            "mean_cost_usd": p.mean_cost_usd,
        }
        for p in points
    ]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """A single run's full per-case results and scores (drill-down view)."""
    logger.info(
        "GET /runs/{run_id}",
        extra={"event": "api_request", "route": "/runs/{run_id}", "run_id": run_id},
    )
    try:
        run = load_run(run_id, runs_dir=_runs_dir())
    except FileNotFoundError as e:
        logger.warning(
            str(e),
            extra={
                "event": "api_error",
                "route": "/runs/{run_id}",
                "status_code": 404,
                "run_id": run_id,
            },
        )
        raise HTTPException(status_code=404, detail=str(e)) from e
    return run.model_dump(mode="json")


@app.get("/compare/latest")
def get_latest_comparison() -> dict:
    """Comparison between the two most recently persisted runs."""
    logger.info(
        "GET /compare/latest",
        extra={"event": "api_request", "route": "/compare/latest"},
    )
    runs = load_runs(runs_dir=_runs_dir())
    if len(runs) < 2:
        logger.warning(
            "not enough persisted runs to compare",
            extra={
                "event": "api_error",
                "route": "/compare/latest",
                "status_code": 404,
            },
        )
        raise HTTPException(
            status_code=404, detail="need at least 2 persisted runs to compare"
        )
    baseline, candidate = runs[-2], runs[-1]
    try:
        report = compare_runs(
            baseline.results, baseline.scores, candidate.results, candidate.scores
        )
    except ValueError as e:
        logger.warning(
            str(e),
            extra={
                "event": "api_error",
                "route": "/compare/latest",
                "status_code": 400,
            },
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        **_report_dict(report),
    }


@app.get("/compare/{baseline_run_id}/{candidate_run_id}")
def get_comparison(baseline_run_id: str, candidate_run_id: str) -> dict:
    """Comparison between two explicitly-named persisted runs."""
    logger.info(
        "GET /compare/{baseline_run_id}/{candidate_run_id}",
        extra={
            "event": "api_request",
            "route": "/compare/{baseline_run_id}/{candidate_run_id}",
            "baseline_run_id": baseline_run_id,
            "candidate_run_id": candidate_run_id,
        },
    )
    try:
        baseline = load_run(baseline_run_id, runs_dir=_runs_dir())
        candidate = load_run(candidate_run_id, runs_dir=_runs_dir())
    except FileNotFoundError as e:
        logger.warning(
            str(e),
            extra={
                "event": "api_error",
                "route": "/compare/{baseline_run_id}/{candidate_run_id}",
                "status_code": 404,
            },
        )
        raise HTTPException(status_code=404, detail=str(e)) from e
    try:
        report = compare_runs(
            baseline.results, baseline.scores, candidate.results, candidate.scores
        )
    except ValueError as e:
        logger.warning(
            str(e),
            extra={
                "event": "api_error",
                "route": "/compare/{baseline_run_id}/{candidate_run_id}",
                "status_code": 400,
            },
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        **_report_dict(report),
    }
