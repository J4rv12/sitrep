"""One log line = one JSON object. CLAUDE.md section 6.

Stdlib `logging` plus a formatter, not structlog: the requirement is "every
line parses as JSON and carries the same keys", which is a screenful of
code, not a dependency.

After the 202 is sent, a log line is the only channel left — Phase 1 showed
an exception in a background task reaches nobody but stderr. So the fields
every event must carry are keyword-only arguments to `log_event`, and
forgetting one is a TypeError at the call site rather than a gap in the logs.
"""

import json
import logging

_EVENT_FIELDS = ("alert_id", "stage", "outcome", "latency_ms", "reason")


class JsonFormatter(logging.Formatter):
    """Render one record as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {"level": record.levelname, "msg": record.getMessage()}
        for field in _EVENT_FIELDS:
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str) -> None:
    """Route every app log record to stderr as JSON.

    Replaces the root logger's handlers rather than appending, so calling it
    twice does not print every line twice. Raises `ValueError` on an unknown
    level name — called from `main.py` at import, so a typo in LOG_LEVEL
    fails the deploy instead of silently logging at the wrong level.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # httpx logs every request URL at INFO. Telegram's Bot API carries the
    # bot token in the URL, so at INFO every delivery would write the token
    # into Render's logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # The anthropic SDK sends through httpx2, a separate package with its own
    # logger. Its URLs carry no secret, but each request would add a line with
    # none of the event fields, one per Claude call.
    logging.getLogger("httpx2").setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger,
    *,
    alert_id: str,
    stage: str,
    outcome: str,
    latency_ms: float,
    reason: str | None = None,
    exc_info: bool = False,
) -> None:
    """Log one pipeline event. WARNING when it carries a reason code, else INFO.

    `exc_info=True` attaches the exception being handled and raises the
    level to ERROR. Only for the last-resort net in `process_alert`: an
    expected failure has a reason code and needs no traceback, and a line
    that says `internal_error` without one hides the bug it caught.
    """
    fields: dict[str, object] = {
        "alert_id": alert_id,
        "stage": stage,
        "outcome": outcome,
        "latency_ms": round(latency_ms, 1),
    }
    if reason is not None:
        fields["reason"] = reason
    level = logging.INFO if reason is None else logging.WARNING
    if exc_info:
        level = logging.ERROR
    logger.log(level, "%s %s", stage, outcome, extra=fields, exc_info=exc_info)
