from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from litmus.schemas import RunResult, RunTarget, TestCase


def test_test_case_requires_id_and_input():
    case = TestCase(id="c1", input="hello")
    assert case.id == "c1"
    assert case.input == "hello"
    assert case.expected_output is None
    assert case.rubric is None
    assert case.tags == []
    assert case.scorer == "exact_match"


def test_test_case_missing_input_raises():
    with pytest.raises(ValidationError):
        TestCase(id="c1")


def test_run_target_fields():
    target = RunTarget(prompt_version="v1", model_name="gpt-4o-mini")
    assert target.prompt_version == "v1"
    assert target.model_name == "gpt-4o-mini"


def test_run_result_fields():
    result = RunResult(
        test_case_id="c1",
        raw_output="positive",
        latency_ms=123.4,
        cost_usd=0.0001,
        timestamp=datetime.now(UTC),
    )
    assert result.test_case_id == "c1"
    assert result.raw_output == "positive"
    assert result.latency_ms == pytest.approx(123.4)
    assert result.cost_usd == pytest.approx(0.0001)
