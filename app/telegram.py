"""An alert and its brief in, a Telegram message out. The last stage.

Two functions, one direction of travel:

    format_message  pure: the text the chat shows, enriched or not
    send_message    all the I/O, and every way Telegram can refuse it

Nothing runs after this stage, and it runs after `process_alert`'s net, not
inside it: when the net turns a bug into None, None still has to ship as
`[unenriched]` (invariant 4). So nothing here may raise. `format_message`
works for every valid `SignalBrief`, and `send_message` turns every failure
into False plus a reason code, the way `get_market_context` does.

No retry. If Telegram accepts a message and the response is then lost, we see
a timeout that looks exactly like a failure, and a retry delivers the alert
twice, which is the defect invariant 5 exists to prevent. The caller records a
failed send; it is not attempted again.
"""

import html
import logging
import time

import httpx

from app.config import get_settings
from app.logs import log_event
from app.schemas import AlertPayload, SignalBrief

logger = logging.getLogger(__name__)

# Telegram's own ceiling, counted after entities like &lt; are parsed.
MAX_MESSAGE_CHARS = 4096


def _escape(text: str) -> str:
    # Telegram's HTML mode needs exactly <, > and & escaped, and supports the
    # entities html.escape produces for them. quote=False: quotes need nothing.
    return html.escape(text, quote=False)


def format_message(payload: AlertPayload, brief: SignalBrief | None) -> str:
    """Return the message for one alert, in Telegram's HTML parse mode.

    With a brief (the first line and the headline are bold):

        BTCUSDT · 1h · severity: low
        close below 20MA

        BTC closes below 1h 20MA on light daily volume at 0.8x average.
        • Price is 0.9% below the 20-day moving average, ...
        • Daily volume is 0.8x the 20-day average, ...

        Bar 2026-09-10T10:00:00+00:00 · price 64000.0

    Without one, the alert as received, tagged per invariant 4:

        [unenriched] BTCUSDT · 1h
        close below 20MA

        Bar 2026-09-10T10:00:00+00:00 · price 64000.0

    The price appears only when the alert carried one.

    Every piece of text is escaped, the model's included. `close < 20MA` is an
    ordinary condition; unescaped, Telegram reads the `<` as the start of a tag,
    fails to parse it, answers 400, and the alert never arrives.

    `bar_time` is shown as the sender gave it, not converted to UTC:
    converting can overflow at the edges of the calendar, and nothing here may
    raise.

    The longest message the schemas allow is about 1,300 characters, well under
    `MAX_MESSAGE_CHARS`, so nothing is truncated. tests/test_telegram.py checks
    it with every field at its limit.

    Pure: no I/O, no config, no clock.
    """
    header = f"{_escape(payload.symbol)} · {_escape(payload.timeframe)}"
    condition = _escape(payload.condition)
    footer = f"Bar {payload.bar_time.isoformat()}"
    if payload.price is not None:
        footer += f" · price {payload.price}"

    if brief is None:
        return f"[unenriched] <b>{header}</b>\n{condition}\n\n{footer}"

    observations = "\n".join(f"• {_escape(item)}" for item in brief.observations)
    return (
        f"<b>{header} · severity: {brief.severity}</b>\n"
        f"{condition}\n\n"
        f"<b>{_escape(brief.headline)}</b>\n"
        f"{observations}\n\n"
        f"{footer}"
    )


async def send_message(
    text: str,
    alert_id: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bool:
    """Send `text` to the configured chat; return True if Telegram accepted it.

    Never raises. Every way the send can fail becomes False plus one WARNING
    line carrying one of these reason codes:

        telegram_timeout        no response within the client's timeout
        telegram_unreachable    DNS, TLS or connection failure
        telegram_invalid_url    the token holds a character a URL cannot, such
                                as a newline pasted into Render's env settings
        telegram_status_<code>  any non-200: 400 text Telegram could not parse
                                or an unknown chat, 401 or 404 a bad bot token,
                                403 the bot was blocked or never started, 429
                                rate limited, 5xx an outage

    Only the status code is read. The body is never parsed, so there is no
    malformed-response case to handle.

    The bot token is part of the URL path. That is why `configure_logging`
    silences httpx's request log, and why this function logs a reason code and
    nothing else: never the URL, never an exception's text. An httpx status
    error's text, for one, quotes the full URL.

    Logs exactly one line per call, stage "telegram", outcome "ok" or
    "degraded".

    `transport` exists for tests, which pass an `httpx.MockTransport`.
    Production leaves it None.
    """
    settings = get_settings()
    started = time.perf_counter()
    reason: str | None = None

    try:
        async with httpx.AsyncClient(
            base_url=settings.telegram_base_url,
            timeout=settings.http_timeout_seconds,
            transport=transport,
        ) as client:
            response = await client.post(
                f"/bot{settings.telegram_bot_token}/sendMessage",
                json={"chat_id": settings.telegram_chat_id, "text": text, "parse_mode": "HTML"},
            )
        if response.status_code != 200:
            reason = f"telegram_status_{response.status_code}"
    # TimeoutException subclasses TransportError, so it must be caught first.
    except httpx.TimeoutException:
        reason = "telegram_timeout"
    except httpx.TransportError:
        reason = "telegram_unreachable"
    except httpx.InvalidURL:
        reason = "telegram_invalid_url"

    log_event(
        logger,
        alert_id=alert_id,
        stage="telegram",
        outcome="ok" if reason is None else "degraded",
        latency_ms=(time.perf_counter() - started) * 1000,
        reason=reason,
    )
    return reason is None
