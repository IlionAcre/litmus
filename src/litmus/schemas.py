from datetime import datetime

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    __test__ = False  # not a pytest test class, despite the name

    id: str
    input: str
    expected_output: str | None = None
    rubric: str | None = None
    tags: list[str] = Field(default_factory=list)
    scorer: str = "exact_match"


class RunTarget(BaseModel):
    prompt_version: str
    model_name: str


class RunResult(BaseModel):
    test_case_id: str
    raw_output: str
    latency_ms: float
    cost_usd: float
    timestamp: datetime


class ScoreResult(BaseModel):
    test_case_id: str
    passed: bool
    score: float
    explanation: str


class PersistedRun(BaseModel):
    run_id: str
    prompt_version: str
    model_name: str
    created_at: datetime
    results: list[RunResult]
    scores: list[ScoreResult]
