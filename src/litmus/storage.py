import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from litmus.logging_config import LOGGER_NAME
from litmus.schemas import PersistedRun, RunResult, RunTarget, ScoreResult

DEFAULT_RUNS_DIR = Path("runs")
logger = logging.getLogger(LOGGER_NAME)


def save_run(
    target: RunTarget,
    results: list[RunResult],
    scores: list[ScoreResult],
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    run_id: str | None = None,
) -> PersistedRun:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run = PersistedRun(
        run_id=run_id or uuid.uuid4().hex,
        prompt_version=target.prompt_version,
        model_name=target.model_name,
        created_at=datetime.now(UTC),
        results=results,
        scores=scores,
    )

    path = runs_dir / f"{run.run_id}.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(run.model_dump_json(indent=2))
    tmp_path.replace(path)  # atomic rename: no partial-write readers

    logger.info(
        "run saved",
        extra={"event": "run_saved", "run_id": run.run_id, "path": path},
    )
    return run


def load_run(run_id: str, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> PersistedRun:
    path = Path(runs_dir) / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no persisted run {run_id!r} found under {runs_dir}")
    return PersistedRun.model_validate_json(path.read_text())


def load_runs(runs_dir: Path | str = DEFAULT_RUNS_DIR) -> list[PersistedRun]:
    """All persisted runs, ordered chronologically by created_at.

    Deliberately not filename order: run_ids default to a random UUID4 hex,
    which has no relationship to creation time, so callers that want "the
    most recent run" (e.g. the dashboard's latest-comparison route) need a
    real chronological ordering, not glob/filename order.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    runs = [
        PersistedRun.model_validate_json(path.read_text())
        for path in runs_dir.glob("*.json")
    ]
    return sorted(runs, key=lambda r: r.created_at)
