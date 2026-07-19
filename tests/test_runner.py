from litmus.runner import run_test_case
from litmus.schemas import RunTarget, TestCase


def test_run_test_case_produces_correctly_shaped_result():
    case = TestCase(id="c1", input="hello", expected_output="hi")
    target = RunTarget(prompt_version="v1", model_name="stub-model")

    def stub_call_fn(case, target):
        return ("hi", 42.0, 0.0005)

    result = run_test_case(case, target, stub_call_fn)

    assert result.test_case_id == "c1"
    assert result.raw_output == "hi"
    assert result.latency_ms == 42.0
    assert result.cost_usd == 0.0005
    assert result.timestamp is not None


def test_run_test_case_passes_case_and_target_to_call_fn():
    case = TestCase(id="c2", input="foo")
    target = RunTarget(prompt_version="v2", model_name="stub-model-2")
    received = {}

    def stub_call_fn(case, target):
        received["case"] = case
        received["target"] = target
        return ("bar", 1.0, 0.0)

    run_test_case(case, target, stub_call_fn)

    assert received["case"] is case
    assert received["target"] is target
