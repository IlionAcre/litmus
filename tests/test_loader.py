import json

import pytest

from litmus.loader import TestCaseLoadError, load_test_cases


def test_loads_all_valid_test_cases_in_directory(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"id": "a", "input": "hi"}))
    (tmp_path / "b.json").write_text(json.dumps({"id": "b", "input": "yo"}))

    cases = load_test_cases(tmp_path)

    assert {c.id for c in cases} == {"a", "b"}


def test_loads_the_example_testset():
    cases = load_test_cases("testsets/example")

    assert len(cases) == 2
    assert {c.id for c in cases} == {"case_001", "case_002"}


def test_malformed_json_raises_clear_error(tmp_path):
    (tmp_path / "broken.json").write_text("{not valid json")

    with pytest.raises(TestCaseLoadError, match="broken.json"):
        load_test_cases(tmp_path)


def test_schema_violation_raises_clear_error(tmp_path):
    (tmp_path / "bad_schema.json").write_text(json.dumps({"id": "x"}))

    with pytest.raises(TestCaseLoadError, match="bad_schema.json"):
        load_test_cases(tmp_path)
