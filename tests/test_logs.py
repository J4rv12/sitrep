"""The log contract: one JSON object per line, and no request URLs in it.

After the 202 is sent, a log line is the only record an alert leaves. So the
shape is a contract, asserted with full equality, like `/healthz`'s body.
"""

import io
import json
import logging

from app.logs import JsonFormatter, configure_logging, log_event


def test_event_is_one_json_line_carrying_the_required_fields() -> None:
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("tests.logs")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        log_event(
            logger,
            alert_id="alert-1",
            stage="enrich",
            outcome="degraded",
            latency_ms=12.34,
            reason="binance_timeout",
        )
    finally:
        logger.removeHandler(handler)

    (line,) = buffer.getvalue().splitlines()
    assert json.loads(line) == {
        "level": "WARNING",
        "msg": "enrich degraded",
        "alert_id": "alert-1",
        "stage": "enrich",
        "outcome": "degraded",
        "latency_ms": 12.3,
        "reason": "binance_timeout",
    }


def test_unexpected_failure_logs_the_traceback_at_error() -> None:
    """The last-resort net's line must say what it caught, or it hides bugs."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("tests.logs")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        try:
            raise ZeroDivisionError("simulated")
        except ZeroDivisionError:
            log_event(
                logger,
                alert_id="alert-1",
                stage="pipeline",
                outcome="degraded",
                latency_ms=1.0,
                reason="internal_error",
                exc_info=True,
            )
    finally:
        logger.removeHandler(handler)

    (line,) = buffer.getvalue().splitlines()
    entry = json.loads(line)
    assert entry["level"] == "ERROR"
    assert entry["reason"] == "internal_error"
    assert "ZeroDivisionError: simulated" in entry["exc"]
    assert "Traceback" in entry["exc"]


def test_httpx_request_urls_stay_out_of_the_logs() -> None:
    """Telegram's Bot API puts the bot token in the URL, and httpx logs URLs at INFO.

    Drop the httpx line from `configure_logging` and this fails, because the
    root logger is at INFO and httpx inherits it.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("INFO")
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
