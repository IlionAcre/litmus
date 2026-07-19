from litmus.scoring.base import Scorer
from litmus.scoring.exact_match import ExactMatchScorer, JsonSchemaMatchScorer
from litmus.scoring.llm_judge import LlmJudgeScorer
from litmus.scoring.semantic_similarity import SemanticSimilarityScorer

SCORERS: dict[str, Scorer] = {
    "exact_match": ExactMatchScorer(),
    "json_schema_match": JsonSchemaMatchScorer(),
    "semantic_similarity": SemanticSimilarityScorer(),
    "llm_judge": LlmJudgeScorer(),
}


def get_scorer(name: str) -> Scorer:
    try:
        return SCORERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown scorer {name!r}. Available: {sorted(SCORERS)}"
        ) from None
