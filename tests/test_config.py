from pathlib import Path

import pytest

from litmus.config import ConfigError, LitmusConfig, load_config


def test_load_config_falls_back_to_defaults_with_no_pyproject_toml(tmp_path):
    config = load_config(start_dir=tmp_path)

    assert config == LitmusConfig()


def test_load_config_falls_back_to_defaults_with_no_tool_litmus_section(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"other\"\n")

    config = load_config(start_dir=tmp_path)

    assert config == LitmusConfig()


def test_load_config_overrides_only_the_keys_present(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.litmus]\nalpha = 0.01\nmin_case_count = 20\n"
    )

    config = load_config(start_dir=tmp_path)

    assert config.alpha == 0.01
    assert config.min_case_count == 20
    # everything else stays default
    assert config.confidence == LitmusConfig().confidence
    assert config.max_workers == LitmusConfig().max_workers


def test_load_config_overrides_scorer_settings(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.litmus.scorers.llm_judge]\nthreshold = 0.7\nmodel = \"gemini/custom\"\n"
    )

    config = load_config(start_dir=tmp_path)

    assert config.llm_judge_threshold == 0.7
    assert config.llm_judge_model == "gemini/custom"
    # the other scorer's settings are untouched
    assert config.semantic_similarity_threshold == LitmusConfig().semantic_similarity_threshold


def test_load_config_converts_path_fields(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.litmus]\nruns_dir = "custom_runs"\nlog_file = "custom/log.jsonl"\n'
    )

    config = load_config(start_dir=tmp_path)

    assert config.runs_dir == Path("custom_runs")
    assert config.log_file == Path("custom/log.jsonl")


@pytest.mark.parametrize("field_name,value", [("alpha", 1.5), ("confidence", -0.1)])
def test_load_config_rejects_out_of_range_unit_interval_fields(tmp_path, field_name, value):
    (tmp_path / "pyproject.toml").write_text(f"[tool.litmus]\n{field_name} = {value}\n")

    with pytest.raises(ConfigError, match=f"{field_name} = {value!r}"):
        load_config(start_dir=tmp_path)


def test_load_config_rejects_negative_int_fields(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.litmus]\nmax_workers = -1\n")

    with pytest.raises(ConfigError, match="max_workers"):
        load_config(start_dir=tmp_path)


def test_config_error_names_pyproject_toml_not_a_cli_flag(tmp_path):
    """The whole point of validating here instead of relying on Typer's
    min=/max= alone: a bad config-file value must not produce a misleading
    "Invalid value for '--alpha'"-style message blaming a flag the user
    never touched."""
    (tmp_path / "pyproject.toml").write_text("[tool.litmus]\nalpha = 2.0\n")

    with pytest.raises(ConfigError, match="pyproject.toml"):
        load_config(start_dir=tmp_path)
