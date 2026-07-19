from datetime import UTC, datetime

from litmus.schemas import RunResult, ScoreResult, TestCase
from litmus.scoring.base import Scorer


class DummyScorer:
    """Trivial Scorer implementation used only to prove the interface out."""

    def score(self, case: TestCase, result: RunResult) -> ScoreResult:
        passed = result.raw_output == case.expected_output
        return ScoreResult(
            test_case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            explanation="dummy scorer: exact equality check",
        )


def _run_result(test_case_id: str, raw_output: str) -> RunResult:
    return RunResult(
        test_case_id=test_case_id,
        raw_output=raw_output,
        latency_ms=1.0,
        cost_usd=0.0,
        timestamp=datetime.now(UTC),
    )


def test_dummy_scorer_satisfies_scorer_protocol():
    scorer: Scorer = DummyScorer()
    case = TestCase(id="c1", input="hi", expected_output="hello")
    result = _run_result("c1", "hello")

    score_result = scorer.score(case, result)

    assert score_result.test_case_id == "c1"
    assert score_result.passed is True
    assert score_result.score == 1.0


def test_dummy_scorer_fails_on_mismatch():
    scorer = DummyScorer()
    case = TestCase(id="c2", input="hi", expected_output="hello")
    result = _run_result("c2", "goodbye")

    score_result = scorer.score(case, result)

    assert score_result.passed is False
    assert score_result.score == 0.0
