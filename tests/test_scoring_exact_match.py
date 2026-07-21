from datetime import UTC, datetime

import pytest

from litmus.schemas import RunResult, TestCase
from litmus.scoring.exact_match import ExactMatchScorer, JsonSchemaMatchScorer
from litmus.scoring.registry import get_scorer


def _result(raw_output: str) -> RunResult:
    return RunResult(
        test_case_id="c1",
        raw_output=raw_output,
        latency_ms=1.0,
        cost_usd=0.0,
        timestamp=datetime.now(UTC),
    )


def test_exact_match_passes_on_equal_output():
    scorer = ExactMatchScorer()
    case = TestCase(id="c1", input="x", expected_output="positive")

    score_result = scorer.score(case, _result("positive"))

    assert score_result.passed is True
    assert score_result.score == 1.0


def test_exact_match_fails_on_different_output():
    scorer = ExactMatchScorer()
    case = TestCase(id="c1", input="x", expected_output="positive")

    score_result = scorer.score(case, _result("negative"))

    assert score_result.passed is False
    assert score_result.score == 0.0


def test_exact_match_strips_surrounding_whitespace():
    scorer = ExactMatchScorer()
    case = TestCase(id="c1", input="x", expected_output="positive")

    score_result = scorer.score(case, _result("  positive  \n"))

    assert score_result.passed is True


def test_exact_match_requires_expected_output():
    scorer = ExactMatchScorer()
    case = TestCase(id="c1", input="x")

    with pytest.raises(ValueError, match="expected_output"):
        scorer.score(case, _result("positive"))


def test_json_schema_match_passes_on_equivalent_json_regardless_of_key_order():
    scorer = JsonSchemaMatchScorer()
    case = TestCase(id="c1", input="x", expected_output='{"a": 1, "b": 2}')

    score_result = scorer.score(case, _result('{"b": 2, "a": 1}'))

    assert score_result.passed is True


def test_json_schema_match_fails_on_invalid_json_output():
    scorer = JsonSchemaMatchScorer()
    case = TestCase(id="c1", input="x", expected_output='{"a": 1}')

    score_result = scorer.score(case, _result("not json"))

    assert score_result.passed is False


def test_json_schema_match_fails_on_structurally_different_json():
    scorer = JsonSchemaMatchScorer()
    case = TestCase(id="c1", input="x", expected_output='{"a": 1}')

    score_result = scorer.score(case, _result('{"a": 2}'))

    assert score_result.passed is False


def test_json_schema_match_requires_expected_output():
    scorer = JsonSchemaMatchScorer()
    case = TestCase(id="c1", input="x")

    with pytest.raises(ValueError, match="expected_output"):
        scorer.score(case, _result('{"a": 1}'))


def test_json_schema_match_raises_when_expected_output_itself_is_invalid_json():
    """Previously claimed (but never actually tested) to fail loudly - this
    proves it, rather than just asserting the code reads that way."""
    scorer = JsonSchemaMatchScorer()
    case = TestCase(id="c1", input="x", expected_output="{not valid json")

    with pytest.raises(ValueError, match="expected_output to be valid JSON"):
        scorer.score(case, _result('{"a": 1}'))


def test_registry_looks_up_scorer_by_name():
    assert isinstance(get_scorer("exact_match"), ExactMatchScorer)
    assert isinstance(get_scorer("json_schema_match"), JsonSchemaMatchScorer)


def test_registry_raises_clear_error_on_unknown_scorer():
    with pytest.raises(ValueError, match="Unknown scorer"):
        get_scorer("nonexistent")
