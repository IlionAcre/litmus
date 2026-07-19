from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from litmus.schemas import RunResult, TestCase
from litmus.scoring.llm_judge import JudgeParseError, LlmJudgeScorer
from litmus.scoring.registry import get_scorer


def _result(raw_output: str) -> RunResult:
    return RunResult(
        test_case_id="c1",
        raw_output=raw_output,
        latency_ms=1.0,
        cost_usd=0.0,
        timestamp=datetime.now(UTC),
    )


def _fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_judge_passes_on_valid_json_verdict(monkeypatch):
    scorer = LlmJudgeScorer()
    case = TestCase(
        id="c1", input="x", rubric="Output must be polite and mention a refund."
    )

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response(
            '{"passed": true, "rationale": "polite and mentions refund"}'
        ),
    )

    score_result = scorer.score(case, _result("Sure, here's your refund!"))

    assert score_result.passed is True
    assert score_result.score == 1.0
    assert score_result.explanation == "polite and mentions refund"


def test_judge_fails_on_valid_json_verdict_marking_failure(monkeypatch):
    scorer = LlmJudgeScorer()
    case = TestCase(id="c1", input="x", rubric="Must mention a refund.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response(
            '{"passed": false, "rationale": "no refund mentioned"}'
        ),
    )

    score_result = scorer.score(case, _result("Have a nice day."))

    assert score_result.passed is False
    assert score_result.score == 0.0


def test_judge_raises_clear_error_on_unparseable_response(monkeypatch):
    """A judge that returns garbage must fail loudly, not be silently
    coerced into a pass/fail."""
    scorer = LlmJudgeScorer()
    case = TestCase(id="c1", input="x", rubric="Must be polite.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response(
            "I think this passes, looks good to me!"
        ),
    )

    with pytest.raises(JudgeParseError, match="could not be parsed"):
        scorer.score(case, _result("Sure thing."))


def test_judge_raises_clear_error_on_json_missing_required_fields(monkeypatch):
    scorer = LlmJudgeScorer()
    case = TestCase(id="c1", input="x", rubric="Must be polite.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response('{"verdict": "yes"}'),
    )

    with pytest.raises(JudgeParseError):
        scorer.score(case, _result("Sure thing."))


def test_judge_requires_rubric():
    scorer = LlmJudgeScorer()
    case = TestCase(id="c1", input="x")

    with pytest.raises(ValueError, match="rubric"):
        scorer.score(case, _result("anything"))


def test_registered_under_llm_judge():
    assert isinstance(get_scorer("llm_judge"), LlmJudgeScorer)
