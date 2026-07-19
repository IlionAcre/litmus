import time

import litellm

from litmus.schemas import RunTarget, TestCase


def litellm_call(case: TestCase, target: RunTarget) -> tuple[str, float, float]:
    start = time.perf_counter()
    response = litellm.completion(
        model=target.model_name,
        messages=[{"role": "user", "content": case.input}],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    output = response.choices[0].message.content
    cost_usd = litellm.completion_cost(completion_response=response)
    return output, latency_ms, cost_usd
