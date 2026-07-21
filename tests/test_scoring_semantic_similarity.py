from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from litmus.schemas import RunResult, TestCase
from litmus.scoring.registry import get_scorer
from litmus.scoring.semantic_similarity import SemanticSimilarityScorer, _cosine_similarity


def _result(raw_output: str) -> RunResult:
    return RunResult(
        test_case_id="c1",
        raw_output=raw_output,
        latency_ms=1.0,
        cost_usd=0.0,
        timestamp=datetime.now(UTC),
    )


def _fake_embedding_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [{"embedding": v} for v in vectors]
    return response


def test_passes_on_near_identical_embeddings(monkeypatch):
    scorer = SemanticSimilarityScorer(threshold=0.8)
    case = TestCase(id="c1", input="x", expected_output="The cat sat on the mat")

    monkeypatch.setattr(
        "litellm.embedding",
        lambda model, input: _fake_embedding_response(
            [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0]]
        ),
    )

    score_result = scorer.score(case, _result("A cat was sitting on the mat"))

    assert score_result.passed is True
    assert score_result.score > 0.9


def test_fails_on_orthogonal_embeddings(monkeypatch):
    scorer = SemanticSimilarityScorer(threshold=0.8)
    case = TestCase(id="c1", input="x", expected_output="The cat sat on the mat")

    monkeypatch.setattr(
        "litellm.embedding",
        lambda model, input: _fake_embedding_response([[1.0, 0.0], [0.0, 1.0]]),
    )

    score_result = scorer.score(case, _result("Stock prices rose sharply today"))

    assert score_result.passed is False
    assert score_result.score < 0.8


def test_requires_expected_output():
    scorer = SemanticSimilarityScorer()
    case = TestCase(id="c1", input="x")

    with pytest.raises(ValueError, match="expected_output"):
        scorer.score(case, _result("anything"))


def test_registered_under_semantic_similarity(monkeypatch):
    scorer = get_scorer("semantic_similarity")
    assert isinstance(scorer, SemanticSimilarityScorer)


def test_cosine_similarity_zero_norm_guard_does_not_divide_by_zero():
    """A zero vector (norm 0) must return 0.0 similarity, not raise a
    ZeroDivisionError - defensive code that's never actually been exercised
    by a test before."""
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0
    assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
