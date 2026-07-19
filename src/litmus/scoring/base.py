from typing import Protocol

from litmus.schemas import RunResult, ScoreResult, TestCase


class Scorer(Protocol):
    def score(self, case: TestCase, result: RunResult) -> ScoreResult: ...
