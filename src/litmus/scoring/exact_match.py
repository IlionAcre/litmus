import json

from litmus.schemas import RunResult, ScoreResult, TestCase


class ExactMatchScorer:
    """Passes iff raw_output exactly equals expected_output (whitespace-trimmed)."""

    def score(self, case: TestCase, result: RunResult) -> ScoreResult:
        if case.expected_output is None:
            raise ValueError(
                f"ExactMatchScorer requires expected_output; test case "
                f"{case.id!r} has none"
            )
        passed = result.raw_output.strip() == case.expected_output.strip()
        return ScoreResult(
            test_case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            explanation=(
                "exact match"
                if passed
                else f"expected {case.expected_output!r}, got {result.raw_output!r}"
            ),
        )


class JsonSchemaMatchScorer:
    """Passes iff raw_output parses as JSON and deep-equals expected_output's JSON."""

    def score(self, case: TestCase, result: RunResult) -> ScoreResult:
        if case.expected_output is None:
            raise ValueError(
                f"JsonSchemaMatchScorer requires expected_output; test case "
                f"{case.id!r} has none"
            )
        try:
            actual = json.loads(result.raw_output)
        except json.JSONDecodeError:
            return ScoreResult(
                test_case_id=case.id,
                passed=False,
                score=0.0,
                explanation=f"raw_output is not valid JSON: {result.raw_output!r}",
            )

        try:
            expected = json.loads(case.expected_output)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JsonSchemaMatchScorer requires expected_output to be valid "
                f"JSON for test case {case.id!r}"
            ) from e

        passed = actual == expected
        return ScoreResult(
            test_case_id=case.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            explanation=(
                "json match" if passed else f"expected {expected!r}, got {actual!r}"
            ),
        )
