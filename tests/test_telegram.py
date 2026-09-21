"""Delivery: the message the chat shows, and every way Telegram can refuse it.

The brief is real: the text of the recorded Claude reply tests/test_llm.py
uses, for the same alert. Text written for a test — the escaping case, the
fields at their length limits, the broken token — says so where it appears.

No recorded Telegram responses, on purpose. `send_message` reads only the
status code, so a body would be decoration, and a real sendMessage reply
carries the chat's id and the account's name, which don't belong in a public
repo. Statuses are served with empty bodies; network failures are simulated at
the transport, as in tests/test_enrich.py.
"""

import html
import json
import logging
import re
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from app import telegram
from app.config import Settings, get_settings
from app.logs import JsonFormatter
from app.schemas import AlertPayload, SignalBrief
from app.telegram import MAX_MESSAGE_CHARS, format_message, send_message

RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "anthropic_brief_btcusdt_low_volume.json").read_bytes()
)
BRIEF = SignalBrief.model_validate_json(RECORDED["content"][0]["text"])
ALERT = AlertPayload(
    symbol="BTCUSDT",
    timeframe="1h",
    condition="close below 20MA",
    bar_time="2026-09-10T10:00:00Z",
)


def displayed(text: str) -> str:
    """What the chat shows: the formatter's <b> tags gone, entities turned back."""
    return html.unescape(re.sub(r"</?b>", "", text))


# --- format_message ---------------------------------------------------------


def test_brief_becomes_the_message() -> None:
    assert format_message(ALERT, BRIEF) == (
        "<b>BTCUSDT · 1h · severity: low</b>\n"
        "close below 20MA\n"
        "\n"
        "<b>BTC closes below 1h 20MA on light daily volume at 0.8x average.</b>\n"
        "• Price is 0.9% below the 20-day moving average, showing minimal extension.\n"
        "• Daily volume is 0.8x the 20-day average, indicating weak participation in the move.\n"
        "• The alert fired on a 1-hour timeframe while daily-level context suggests limited "
        "conviction behind downward pressure.\n"
        "\n"
        "Bar 2026-09-10T10:00:00+00:00"
    )


def test_no_brief_ships_the_alert_tagged_unenriched() -> None:
    """Invariant 4: the alert still arrives, and says it arrived without context."""
    assert format_message(ALERT, None) == (
        "[unenriched] <b>BTCUSDT · 1h</b>\nclose below 20MA\n\nBar 2026-09-10T10:00:00+00:00"
    )


@pytest.mark.parametrize("brief", [BRIEF, None], ids=["enriched", "unenriched"])
def test_price_is_shown_when_the_alert_carries_one(brief: SignalBrief | None) -> None:
    text = format_message(ALERT.model_copy(update={"price": 64000.0}), brief)

    assert text.endswith("Bar 2026-09-10T10:00:00+00:00 · price 64000.0")


def test_every_piece_of_text_is_escaped() -> None:
    """An unescaped `<` or `&` is a 400 from Telegram, and the alert never arrives.

    Every text field carries characters HTML mode reserves, the model's
    fields included. Written for this test.
    """
    alert = AlertPayload(
        symbol="BTC<USDT",
        timeframe="1h&4h",
        condition="close < 20MA & volume > average",
        bar_time="2026-09-10T10:00:00Z",
    )
    brief = SignalBrief(
        severity="low",
        headline="Volume < average & price > 20MA",
        observations=["<b>not bold</b>", "volume & price"],
    )

    text = format_message(alert, brief)

    raw = re.sub(r"</?b>", "", text)  # The formatter's own tags are the only markup allowed.
    assert "<" not in raw and ">" not in raw
    assert re.search(r"&(?!lt;|gt;|amp;)", raw) is None, "a bare & that is not an entity"
    for original in [alert.symbol, alert.timeframe, alert.condition, brief.headline]:
        assert original in displayed(text)
    assert "<b>not bold</b>" in displayed(text), "the model's markup must show, not render"


def test_longest_possible_message_fits_telegrams_limit() -> None:
    """Every field at its schema limit, in a character that counts double.

    Telegram counts after parsing entities. Whether it counts characters or
    UTF-16 units, an astral character like this one is the worst case: one
    character, two units. Written for this test.
    """
    wide = "\U0001d400"  # MATHEMATICAL BOLD CAPITAL A, outside the BMP.
    alert = AlertPayload(
        symbol=wide * 32,
        timeframe=wide * 16,
        condition=wide * 200,
        bar_time="2026-09-10T10:00:00.123456+05:30",
        price=1.7976931348623157e308,
    )
    brief = SignalBrief(severity="medium", headline=wide * 120, observations=[wide * 200] * 4)

    shown = displayed(format_message(alert, brief))

    assert len(shown.encode("utf-16-le")) // 2 <= MAX_MESSAGE_CHARS


# --- send_message -----------------------------------------------------------


@pytest.fixture(autouse=True)
def capture_info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.telegram")


def telegram_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "app.telegram"]


def assert_no_token_in(records: list[logging.LogRecord]) -> None:
    """The token is in the URL path, and an httpx status error's text quotes the URL."""
    token = get_settings().telegram_bot_token
    for record in records:
        assert token not in JsonFormatter().format(record)


def serve(status: int, seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    """A transport that answers every request with `status` and an empty body."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status)

    return httpx.MockTransport(handler)


def fail_with(error: type[httpx.TransportError]) -> httpx.MockTransport:
    """A transport that raises the way httpx does when the network fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error("simulated", request=request)

    return httpx.MockTransport(handler)


async def test_accepted_message_returns_true(caplog: pytest.LogCaptureFixture) -> None:
    seen: list[httpx.Request] = []

    assert await send_message("hello", "alert-1", transport=serve(200, seen)) is True

    settings = get_settings()
    (request,) = seen
    assert request.method == "POST"
    assert request.url.host == "api.telegram.org"
    assert request.url.path == f"/bot{settings.telegram_bot_token}/sendMessage"
    assert json.loads(request.content) == {
        "chat_id": settings.telegram_chat_id,
        "text": "hello",
        "parse_mode": "HTML",
    }

    records = telegram_records(caplog)
    (record,) = records
    assert (record.levelname, record.alert_id, record.stage, record.outcome) == (
        "INFO",
        "alert-1",
        "telegram",
        "ok",
    )
    assert_no_token_in(records)


# Not one of these may raise. Delivery runs after process_alert's net, so an
# exception here would reach nobody but stderr, and the alert would be lost.
UPSTREAM_FAILURES = [
    pytest.param(serve(500), "telegram_status_500", id="upstream-500"),
    pytest.param(serve(502), "telegram_status_502", id="bad-gateway-502"),
    pytest.param(serve(429), "telegram_status_429", id="rate-limited"),
    pytest.param(serve(400), "telegram_status_400", id="unparseable-text"),
    pytest.param(serve(401), "telegram_status_401", id="bad-token"),
    pytest.param(serve(403), "telegram_status_403", id="bot-blocked"),
    pytest.param(fail_with(httpx.ReadTimeout), "telegram_timeout", id="read-timeout"),
    pytest.param(fail_with(httpx.ConnectTimeout), "telegram_timeout", id="connect-timeout"),
    pytest.param(fail_with(httpx.ConnectError), "telegram_unreachable", id="unreachable"),
]


@pytest.mark.parametrize(("transport", "reason"), UPSTREAM_FAILURES)
async def test_upstream_failure_degrades_to_false_with_a_reason_code(
    transport: httpx.MockTransport, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    assert await send_message("hello", "alert-1", transport=transport) is False

    records = telegram_records(caplog)
    (record,) = records
    assert record.levelname == "WARNING"
    assert (record.alert_id, record.stage, record.outcome, record.reason) == (
        "alert-1",
        "telegram",
        "degraded",
        reason,
    )
    assert_no_token_in(records)


async def test_a_token_that_cannot_go_in_a_url_degrades(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A newline pasted into Render's env settings, say. Written for this test.

    httpx refuses to build the URL and raises `InvalidURL`, which is not a
    network error, so the network catches alone would let it through.
    """
    broken = get_settings().model_copy(update={"telegram_bot_token": "123456789:abc\ndef"})
    monkeypatch.setattr(telegram, "get_settings", lambda: broken)

    assert await send_message("hello", "alert-1", transport=serve(200)) is False

    (record,) = telegram_records(caplog)
    assert (record.levelname, record.reason) == ("WARNING", "telegram_invalid_url")


@pytest.mark.parametrize("name", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"])
def test_an_empty_telegram_setting_fails_at_boot(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`TELEGRAM_BOT_TOKEN=` in .env reads as "". Accepted, every delivery would fail."""
    monkeypatch.setenv(name, "")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
