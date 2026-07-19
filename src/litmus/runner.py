from datetime import UTC, datetime
from typing import Callable

from litmus.schemas import RunResult, RunTarget, TestCase

CallFn = Callable[[TestCase, RunTarget], tuple[str, float, float]]


def run_test_case(case: TestCase, target: RunTarget, call_fn: CallFn) -> RunResult:
    output, latency_ms, cost_usd = call_fn(case, target)
    return RunResult(
        test_case_id=case.id,
        raw_output=output,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        timestamp=datetime.now(UTC),
    )
