import pytest


@pytest.fixture(autouse=True)
def _isolate_log_file(monkeypatch, tmp_path):
    """Every test gets its own log file under tmp_path. Without this, the
    CLI callback's --log-file default (logs/litmus.jsonl) would make every
    CliRunner.invoke() call in the suite write into the real project's logs/
    directory - this is the env var fallback Click reads when a test doesn't
    pass --log-file explicitly."""
    monkeypatch.setenv("LITMUS_LOG_FILE", str(tmp_path / "litmus-test.jsonl"))
