import json
import logging
from pathlib import Path

from litmus.logging_config import LOGGER_NAME, JsonFormatter, configure_logging


def _make_record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name=LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_produces_valid_json_with_expected_keys():
    record = _make_record(event="test_event", custom_field="custom_value")

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == LOGGER_NAME
    assert payload["message"] == "hello"
    assert payload["event"] == "test_event"
    assert payload["custom_field"] == "custom_value"
    assert "timestamp" in payload


def test_json_formatter_handles_non_json_native_extra_fields():
    """A stray Path (or any non-JSON-native extra field, e.g. a datetime)
    must not crash the log call - default=str is the safety net."""
    record = _make_record(event="run_started", testset_dir=Path("testsets/example"))

    payload = json.loads(JsonFormatter().format(record))

    assert payload["testset_dir"] == str(Path("testsets/example"))


def test_configure_logging_writes_structured_lines_to_file(tmp_path):
    log_file = tmp_path / "litmus.jsonl"
    configure_logging(log_file, "INFO")

    logger = logging.getLogger(LOGGER_NAME)
    logger.info("did a thing", extra={"event": "did_thing", "foo": "bar"})
    for handler in logger.handlers:
        handler.flush()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "did_thing"
    assert payload["foo"] == "bar"


def test_configure_logging_is_idempotent_and_does_not_accumulate_handlers(tmp_path):
    log_file = tmp_path / "litmus.jsonl"
    configure_logging(log_file, "INFO")
    configure_logging(log_file, "INFO")
    configure_logging(log_file, "INFO")

    logger = logging.getLogger(LOGGER_NAME)
    assert len(logger.handlers) == 1

    logger.info("only once", extra={"event": "once"})
    for handler in logger.handlers:
        handler.flush()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 1


def test_configure_logging_actually_rotates(tmp_path):
    log_file = tmp_path / "litmus.jsonl"
    configure_logging(log_file, "INFO", max_bytes=200, backup_count=2)

    logger = logging.getLogger(LOGGER_NAME)
    for i in range(50):
        logger.info("x" * 50, extra={"event": "pad", "i": i})

    assert log_file.exists()
    assert (tmp_path / "litmus.jsonl.1").exists()
