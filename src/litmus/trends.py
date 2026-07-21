from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from litmus.storage import DEFAULT_RUNS_DIR

_RESULT_STRUCT = (
    'STRUCT(test_case_id VARCHAR, raw_output VARCHAR, latency_ms DOUBLE, '
    'cost_usd DOUBLE, "timestamp" TIMESTAMP, error VARCHAR)[]'
)
_SCORE_STRUCT = (
    "STRUCT(test_case_id VARCHAR, passed BOOLEAN, score DOUBLE, "
    "explanation VARCHAR, error VARCHAR)[]"
)

# An explicit schema is required rather than read_json_auto's inference:
# a run with zero test cases (results=[] / scores=[]) makes DuckDB infer a
# generic JSON type for that file's empty array instead of a typed struct
# list, which then fails avg() with "no function matches avg(JSON)" — either
# for that file alone, or for the whole glob once mixed with populated runs.
# Declaring the schema up front sidesteps per-file type inference entirely.
#
# Each subquery excludes errored cases independently, not jointly: a case
# can have scores.error set (scoring failed) while results.error is unset
# (the LLM call itself succeeded) — its latency/cost data is real and
# should stay in those aggregates even though it's excluded from pass_rate.
# Without this, an errored case's sentinel values (passed=False,
# latency_ms=0.0, cost_usd=0.0 — see cli.py's _run_and_score) drag every
# aggregate down, making a run with failures look artificially *better* on
# cost/latency, not just incomplete.
_TREND_QUERY = f"""
    SELECT
        run_id,
        prompt_version,
        model_name,
        created_at,
        (SELECT avg(CASE WHEN s.passed THEN 1.0 ELSE 0.0 END)
         FROM UNNEST(scores) AS t(s) WHERE s.error IS NULL) AS pass_rate,
        (SELECT avg(r.latency_ms)
         FROM UNNEST(results) AS t(r) WHERE r.error IS NULL) AS mean_latency_ms,
        (SELECT avg(r.cost_usd)
         FROM UNNEST(results) AS t(r) WHERE r.error IS NULL) AS mean_cost_usd
    FROM read_json(
        ?,
        columns={{
            run_id: 'VARCHAR',
            prompt_version: 'VARCHAR',
            model_name: 'VARCHAR',
            created_at: 'TIMESTAMP',
            results: '{_RESULT_STRUCT}',
            scores: '{_SCORE_STRUCT}'
        }}
    )
    ORDER BY created_at
"""


@dataclass
class RunTrendPoint:
    run_id: str
    prompt_version: str
    model_name: str
    created_at: datetime
    # None for a run with zero test cases (avg of an empty set is undefined,
    # not zero) — a degenerate but valid case, not an error.
    pass_rate: float | None
    mean_latency_ms: float | None
    mean_cost_usd: float | None


def query_trends(runs_dir: Path | str = DEFAULT_RUNS_DIR) -> list[RunTrendPoint]:
    """Per-run aggregates (pass rate, mean latency, mean cost) across all
    persisted runs, ordered chronologically. Queries the JSON files under
    runs_dir directly via DuckDB — no database server, no separate index to
    keep in sync."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists() or not any(runs_dir.glob("*.json")):
        return []

    pattern = str(runs_dir / "*.json")
    con = duckdb.connect()
    rows = con.execute(_TREND_QUERY, [pattern]).fetchall()

    return [
        RunTrendPoint(
            run_id=run_id,
            prompt_version=prompt_version,
            model_name=model_name,
            created_at=created_at,
            pass_rate=pass_rate,
            mean_latency_ms=mean_latency_ms,
            mean_cost_usd=mean_cost_usd,
        )
        for (
            run_id,
            prompt_version,
            model_name,
            created_at,
            pass_rate,
            mean_latency_ms,
            mean_cost_usd,
        ) in rows
    ]
