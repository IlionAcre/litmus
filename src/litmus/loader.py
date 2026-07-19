import json
from pathlib import Path

from pydantic import ValidationError

from litmus.schemas import TestCase


class TestCaseLoadError(Exception):
    __test__ = False  # not a pytest test class, despite the name


def load_test_cases(directory: str | Path) -> list[TestCase]:
    directory = Path(directory)
    cases = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise TestCaseLoadError(f"{path}: invalid JSON ({e})") from e
        try:
            cases.append(TestCase(**data))
        except ValidationError as e:
            raise TestCaseLoadError(f"{path}: does not match TestCase schema ({e})") from e
    return cases
