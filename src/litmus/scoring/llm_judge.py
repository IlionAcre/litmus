import json

import litellm
from pydantic import BaseModel, ValidationError

from litmus.schemas import RunResult, ScoreResult, TestCase

DEFAULT_MODEL = "gpt-4o-mini"

JUDGE_PROMPT_TEMPLATE = """You are grading whether an AI system's output satisfies a rubric.

Rubric:
{rubric}

AI system's output:
{output}

Respond with ONLY a JSON object of the form:
{{"passed": true or false, "rationale": "one sentence explaining why"}}
"""


class _JudgeVerdict(BaseModel):
    passed: bool
    rationale: str


class JudgeParseError(Exception):
    """Raised when the judge model's response can't be parsed into a verdict.

    Deliberately raised rather than silently coerced into a pass/fail — a
    judge that returns garbage should be a loud failure, not a quiet one.
    """


class LlmJudgeScorer:
    """Rubric-based grading via an LLM judge, for cases a plain similarity
    score can't capture."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def score(self, case: TestCase, result: RunResult) -> ScoreResult:
        if case.rubric is None:
            raise ValueError(
                f"LlmJudgeScorer requires a rubric; test case {case.id!r} has none"
            )
        prompt = JUDGE_PROMPT_TEMPLATE.format(rubric=case.rubric, output=result.raw_output)
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_verdict = response.choices[0].message.content

        try:
            data = json.loads(raw_verdict)
            verdict = _JudgeVerdict(**data)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            raise JudgeParseError(
                f"judge response for test case {case.id!r} could not be "
                f"parsed as a verdict: {raw_verdict!r} ({e})"
            ) from e

        return ScoreResult(
            test_case_id=case.id,
            passed=verdict.passed,
            score=1.0 if verdict.passed else 0.0,
            explanation=verdict.rationale,
        )
