import json

import litellm
from pydantic import BaseModel, Field, ValidationError

from litmus.schemas import RunResult, ScoreResult, TestCase

DEFAULT_MODEL = "gemini/gemini-2.5-flash-lite"
DEFAULT_THRESHOLD = 0.5

JUDGE_PROMPT_TEMPLATE = """You are grading whether an AI system's output satisfies a rubric.

Rubric:
{rubric}

AI system's output:
{output}

Respond with ONLY a JSON object of the form:
{{"score": <float between 0.0 and 1.0, your confidence that the output fully satisfies the rubric>, "rationale": "one sentence explaining why"}}
"""


class _JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class JudgeParseError(Exception):
    """Raised when the judge model's response can't be parsed into a verdict.

    Deliberately raised rather than silently coerced into a pass/fail — a
    judge that returns garbage (or an out-of-range score) should be a loud
    failure, not a quiet one.
    """


class LlmJudgeScorer:
    """Rubric-based grading via an LLM judge, for cases a plain similarity
    score can't capture. Mirrors SemanticSimilarityScorer's pattern: the
    judge emits a single continuous score (its confidence the output
    satisfies the rubric), thresholded into passed/failed - not a separate,
    independently-judged boolean, which would risk self-contradiction
    (e.g. passed=true, confidence=0.2)."""

    def __init__(self, model: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD):
        self.model = model
        self.threshold = threshold

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

        passed = verdict.score >= self.threshold
        return ScoreResult(
            test_case_id=case.id,
            passed=passed,
            score=verdict.score,
            explanation=(
                f"judge score {verdict.score:.4f} (threshold {self.threshold}): "
                f"{verdict.rationale}"
            ),
        )
