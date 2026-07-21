from litmus.config import load_config
from litmus.scoring.base import Scorer
from litmus.scoring.exact_match import ExactMatchScorer, JsonSchemaMatchScorer
from litmus.scoring.llm_judge import LlmJudgeScorer
from litmus.scoring.semantic_similarity import SemanticSimilarityScorer

_CONFIG = load_config()

SCORERS: dict[str, Scorer] = {
    "exact_match": ExactMatchScorer(),
    "json_schema_match": JsonSchemaMatchScorer(),
    "semantic_similarity": SemanticSimilarityScorer(
        model=_CONFIG.semantic_similarity_model,
        threshold=_CONFIG.semantic_similarity_threshold,
    ),
    "llm_judge": LlmJudgeScorer(
        model=_CONFIG.llm_judge_model, threshold=_CONFIG.llm_judge_threshold
    ),
}


def get_scorer(name: str) -> Scorer:
    try:
        return SCORERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown scorer {name!r}. Available: {sorted(SCORERS)}"
        ) from None
