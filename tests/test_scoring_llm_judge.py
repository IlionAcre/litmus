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


def test_judge_passes_on_high_confidence_score(monkeypatch):
    scorer = LlmJudgeScorer()
    case = TestCase(
        id="c1", input="x", rubric="Output must be polite and mention a refund."
    )

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response(
            '{"score": 0.9, "rationale": "polite and mentions refund"}'
        ),
    )

    score_result = scorer.score(case, _result("Sure, here's your refund!"))

    assert score_result.passed is True
    assert score_result.score == 0.9
    assert score_result.explanation == (
        "judge score 0.9000 (threshold 0.5): polite and mentions refund"
    )


def test_judge_fails_on_low_confidence_score(monkeypatch):
    scorer = LlmJudgeScorer()
    case = TestCase(id="c1", input="x", rubric="Must mention a refund.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response(
            '{"score": 0.1, "rationale": "no refund mentioned"}'
        ),
    )

    score_result = scorer.score(case, _result("Have a nice day."))

    assert score_result.passed is False
    assert score_result.score == 0.1


def test_judge_score_exactly_at_threshold_passes(monkeypatch):
    """Threshold comparison is >=, not >, matching SemanticSimilarityScorer's
    exact convention."""
    scorer = LlmJudgeScorer(threshold=0.5)
    case = TestCase(id="c1", input="x", rubric="Must be polite.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response('{"score": 0.5, "rationale": "borderline"}'),
    )

    score_result = scorer.score(case, _result("..."))

    assert score_result.passed is True


def test_judge_respects_non_default_threshold(monkeypatch):
    scorer = LlmJudgeScorer(threshold=0.9)
    case = TestCase(id="c1", input="x", rubric="Must be polite.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response('{"score": 0.7, "rationale": "mostly polite"}'),
    )

    score_result = scorer.score(case, _result("..."))

    assert score_result.passed is False
    assert score_result.score == 0.7


def test_judge_strips_markdown_code_fence_before_parsing(monkeypatch):
    """Gemini routinely wraps JSON output in a ```json ... ``` code fence
    even when told to respond with ONLY the JSON object - confirmed via a
    live call. Without stripping this, every real judge call fails to parse
    regardless of how well-formed the actual verdict is."""
    scorer = LlmJudgeScorer()
    case = TestCase(id="c1", input="x", rubric="Must be polite.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response(
            '```json\n{"score": 0.8, "rationale": "quite polite"}\n```'
        ),
    )

    score_result = scorer.score(case, _result("Sure thing."))

    assert score_result.score == 0.8
    assert score_result.passed is True


def test_judge_strips_bare_code_fence_without_language_tag(monkeypatch):
    scorer = LlmJudgeScorer()
    case = TestCase(id="c1", input="x", rubric="Must be polite.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response(
            '```\n{"score": 0.2, "rationale": "not polite"}\n```'
        ),
    )

    score_result = scorer.score(case, _result("Sure thing."))

    assert score_result.score == 0.2


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


def test_judge_raises_clear_error_on_out_of_range_score(monkeypatch):
    """A judge score outside [0.0, 1.0] must fail loudly (Field(ge=0.0,
    le=1.0) validation), not be silently clamped or accepted."""
    scorer = LlmJudgeScorer()
    case = TestCase(id="c1", input="x", rubric="Must be polite.")

    monkeypatch.setattr(
        "litellm.completion",
        lambda model, messages: _fake_response('{"score": 1.5, "rationale": "very polite"}'),
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
