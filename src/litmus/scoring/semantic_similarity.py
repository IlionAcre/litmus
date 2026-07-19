import math

import litellm

from litmus.schemas import RunResult, ScoreResult, TestCase

DEFAULT_MODEL = "gemini/text-embedding-004"
DEFAULT_THRESHOLD = 0.8


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticSimilarityScorer:
    """Passes iff the cosine similarity between raw_output and
    expected_output embeddings (via litellm.embedding() — see CLAUDE.md, no
    local sentence-transformers model) meets a threshold."""

    def __init__(self, model: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD):
        self.model = model
        self.threshold = threshold

    def score(self, case: TestCase, result: RunResult) -> ScoreResult:
        if case.expected_output is None:
            raise ValueError(
                f"SemanticSimilarityScorer requires expected_output; test "
                f"case {case.id!r} has none"
            )
        response = litellm.embedding(
            model=self.model,
            input=[result.raw_output, case.expected_output],
        )
        actual_vec = response.data[0]["embedding"]
        expected_vec = response.data[1]["embedding"]
        similarity = _cosine_similarity(actual_vec, expected_vec)
        passed = similarity >= self.threshold
        return ScoreResult(
            test_case_id=case.id,
            passed=passed,
            score=similarity,
            explanation=(
                f"cosine similarity {similarity:.4f} (threshold {self.threshold})"
            ),
        )
