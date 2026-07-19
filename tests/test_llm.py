from unittest.mock import MagicMock

from litmus.llm import litellm_call
from litmus.runner import run_test_case
from litmus.schemas import RunTarget, TestCase


def _fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_litellm_call_returns_output_latency_and_cost(monkeypatch):
    case = TestCase(id="c1", input="hello")
    target = RunTarget(prompt_version="v1", model_name="gpt-4o-mini")
    fake_response = _fake_response("hi there")

    def fake_completion(model, messages):
        assert model == "gpt-4o-mini"
        assert messages == [{"role": "user", "content": "hello"}]
        return fake_response

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr(
        "litellm.completion_cost",
        lambda completion_response: 0.0002,
    )

    output, latency_ms, cost_usd = litellm_call(case, target)

    assert output == "hi there"
    assert latency_ms >= 0
    assert cost_usd == 0.0002


def test_litellm_call_plugs_into_run_test_case_unmodified(monkeypatch):
    case = TestCase(id="c2", input="foo")
    target = RunTarget(prompt_version="v1", model_name="gpt-4o-mini")
    fake_response = _fake_response("bar")

    monkeypatch.setattr(
        "litellm.completion", lambda model, messages: fake_response
    )
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.001
    )

    result = run_test_case(case, target, litellm_call)

    assert result.test_case_id == "c2"
    assert result.raw_output == "bar"
    assert result.cost_usd == 0.001
