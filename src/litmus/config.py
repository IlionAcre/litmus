"""Consolidates every scattered CLI/threshold/scorer default into one place.

Precedence is CLI flag > [tool.litmus] in pyproject.toml > hardcoded
fallback below. Config-file resolution is deliberately a CLI-layer concern
only - stats.py/compare.py/the scorer classes stay pure and config-unaware;
only cli.py and scoring/registry.py read from this module (see CLAUDE.md)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

_PYPROJECT_FILENAME = "pyproject.toml"


class ConfigError(Exception):
    """Raised when pyproject.toml's [tool.litmus] section has an invalid
    value. Deliberately validated here rather than left to Typer's own
    min=/max= option constraints: those apply to a *default* value the same
    as a user-supplied one, but the resulting error message blames the CLI
    flag (e.g. "Invalid value for '--alpha'") even when the user never
    touched that flag and the bad value came from the config file - this
    exception names pyproject.toml and the offending key directly instead."""


@dataclass(frozen=True)
class LitmusConfig:
    runs_dir: Path = Path("runs")
    log_file: Path = Path("logs/litmus.jsonl")
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3
    host: str = "127.0.0.1"
    port: int = 8000
    alpha: float = 0.05
    confidence: float = 0.95
    min_case_count: int = 10
    min_discordant_pairs: int = 10
    exact_threshold: int = 25
    n_resamples: int = 10000
    max_workers: int = 4
    semantic_similarity_model: str = "gemini/text-embedding-004"
    semantic_similarity_threshold: float = 0.8
    llm_judge_model: str = "gemini/gemini-2.5-flash-lite"
    llm_judge_threshold: float = 0.5


_UNIT_INTERVAL_FIELDS = ("alpha", "confidence")
_NON_NEGATIVE_INT_FIELDS = (
    "min_case_count",
    "min_discordant_pairs",
    "exact_threshold",
    "n_resamples",
    "max_workers",
    "port",
)


def _validate(config: LitmusConfig) -> None:
    for field_name in _UNIT_INTERVAL_FIELDS:
        value = getattr(config, field_name)
        if not 0.0 <= value <= 1.0:
            raise ConfigError(
                f"pyproject.toml's [tool.litmus] has {field_name} = {value!r}, "
                "which must be between 0.0 and 1.0"
            )
    for field_name in _NON_NEGATIVE_INT_FIELDS:
        value = getattr(config, field_name)
        if value < 0:
            raise ConfigError(
                f"pyproject.toml's [tool.litmus] has {field_name} = {value!r}, "
                "which must not be negative"
            )


def load_config(start_dir: Path | None = None) -> LitmusConfig:
    """Load [tool.litmus] from pyproject.toml in `start_dir` (default: the
    current working directory - litmus is documented as run from the
    project root, so this does not walk up parent directories). A missing
    file, missing [tool.litmus] section, or any missing key all fall back
    to LitmusConfig's own hardcoded default for that field - no override is
    ever required."""
    pyproject_path = (start_dir or Path.cwd()) / _PYPROJECT_FILENAME
    values: dict = {}

    if pyproject_path.exists():
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        litmus_section = data.get("tool", {}).get("litmus", {})
        scorers_section = litmus_section.get("scorers", {})

        for key, value in litmus_section.items():
            if key != "scorers":
                values[key] = value

        semantic_similarity = scorers_section.get("semantic_similarity", {})
        if "model" in semantic_similarity:
            values["semantic_similarity_model"] = semantic_similarity["model"]
        if "threshold" in semantic_similarity:
            values["semantic_similarity_threshold"] = semantic_similarity["threshold"]

        llm_judge = scorers_section.get("llm_judge", {})
        if "model" in llm_judge:
            values["llm_judge_model"] = llm_judge["model"]
        if "threshold" in llm_judge:
            values["llm_judge_threshold"] = llm_judge["threshold"]

    if "runs_dir" in values:
        values["runs_dir"] = Path(values["runs_dir"])
    if "log_file" in values:
        values["log_file"] = Path(values["log_file"])

    config = LitmusConfig(**values)
    _validate(config)
    return config
