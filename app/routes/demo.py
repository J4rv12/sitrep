"""The public demo: one click, one alert, the pipeline streamed back as it runs.

`POST /demo/{scenario}`, for `breakout`, `weak` or `malformed`. No token: the
page that calls it is static and public, so a secret in it would not be one.
What stands in for the token is `demo_enabled` (off unless a deployment opts
in), the per-IP rate limit, and the daily spend cap.

Why this streams and the webhook does not. TradingView needs to hear "got it"
and never reads the rest, so the webhook answers 202 and works afterwards. A
visitor is waiting on the page for the result. One response that stays open
serves both: its first event goes out before any socket opens, which is
invariant 1 made visible, and the brief follows on the same connection.

    accepted   before anything touches the network
    context    the alert as built, and Binance's two numbers or null
    result     the brief or null, and the exact text Telegram would get

Framed as server-sent events (`text/event-stream`), the content type proxies
are least likely to hold back until the response ends. The page reads it with
`fetch`, not `EventSource`, which reconnects by itself after a dropped
connection and would run the scenario, and bill Claude, a second time.

A demo alert is never delivered or recorded: no dedupe, no Telegram, no feed.
A visitor's click must not message your chat, and must not push your real
alerts out of the feed's 50 entries.

`scan_watchlist`, `get_market_context`, `generate_brief` and
`daily_cap_allows_a_brief` are called by these names so tests/test_demo.py can
replace them, as in routes/webhook.py.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.dedupe import compute_alert_id
from app.enrich import EnrichmentFailed, compute_context, fetch_klines, get_market_context
from app.llm import daily_cap_allows_a_brief, generate_brief
from app.logs import log_event
from app.schemas import AlertPayload, MarketContext
from app.telegram import format_message

router = APIRouter(tags=["demo"])

logger = logging.getLogger(__name__)

SCENARIOS = ("breakout", "weak", "malformed")

# Liquid pairs, each listed for years and still actively traded. Not a list of
# the whole market: the quietest pairs there are the ones being delisted or
# renamed, whose volume is low because trading has stopped, not because a
# signal is weak.
WATCHLIST = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "TRXUSDT", "TONUSDT", "BCHUSDT", "NEARUSDT",
    "UNIUSDT", "SUIUSDT", "ARBUSDT", "ATOMUSDT", "XLMUSDT",
)  # fmt: skip

# A clean breakout is picked from pairs within this distance of their 20MA.
# brief_v1.md calls ~10% stretched but leaves 5-10% undefined, and a 2.2x
# alert at +9% came back medium on 2026-09-22. 5% stays clear of that gap.
CALM_PCT = 5.0

# What a TradingView template sends with `{{exchange}}:{{ticker}}`. It passes
# validation, Binance answers 400, and the alert degrades as a real one would.
MALFORMED_SYMBOL = "BINANCE:BTCUSDT"


async def scan_watchlist(
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[dict[str, MarketContext], str | None]:
    """Fetch every watchlist pair at once; return their contexts and the first failure.

    Pairs Binance cannot supply are left out of the dict. The second value is
    the reason code of the first pair that failed, or None, so a scan where
    every pair failed can say why in its log line.

    Never raises for an upstream failure: `EnrichmentFailed` is caught per
    pair. Anything else is a bug in our code and raises.

    `transport` exists for tests, as in `get_market_context`.
    """
    settings = get_settings()
    async with httpx.AsyncClient(
        base_url=settings.binance_base_url,
        timeout=settings.http_timeout_seconds,
        transport=transport,
    ) as client:
        results = await asyncio.gather(
            *(fetch_klines(symbol, client) for symbol in WATCHLIST), return_exceptions=True
        )

    contexts: dict[str, MarketContext] = {}
    first_failure: str | None = None
    for symbol, result in zip(WATCHLIST, results, strict=True):
        if isinstance(result, EnrichmentFailed):
            first_failure = first_failure or result.reason
        elif isinstance(result, BaseException):
            raise result
        else:
            contexts[symbol] = compute_context(*result)
    return contexts, first_failure


def pick_symbol(scenario: str, contexts: dict[str, MarketContext]) -> str:
    """Return the pair that shows `scenario` best today. `contexts` must not be empty.

    `breakout`: the highest volume among pairs within `CALM_PCT` of their
    20MA, or among all pairs when none is. `weak`: the lowest volume.

    Nothing guarantees the pick earns its label. On a day when every pair
    trades above its average, the quietest is still above it and the brief
    will honestly say medium. That is the demo reading live data, not failing.

    Pure.
    """
    if scenario == "weak":
        return min(contexts, key=lambda symbol: contexts[symbol].volume_vs_20d_avg)
    calm = {s: c for s, c in contexts.items() if abs(c.pct_from_20ma) < CALM_PCT}
    pool = calm or contexts
    return max(pool, key=lambda symbol: pool[symbol].volume_vs_20d_avg)


def demo_alert(symbol: str, context: MarketContext | None) -> AlertPayload:
    """Return the alert a scenario sends, stamped with the current time.

    The condition names the side of the 20MA that `context` puts price on, so
    it is true of the pair it names. Without context there is nothing to
    check it against, and it says "above".

    A fresh `bar_time` per click gives each click its own `alert_id`, so one
    visitor's log lines do not mix with another's.
    """
    side = "below" if context is not None and context.pct_from_20ma < 0 else "above"
    return AlertPayload(
        symbol=symbol,
        timeframe="1h",
        condition=f"close {side} 20MA",
        bar_time=datetime.now(UTC).replace(microsecond=0),
    )


def sse(event: str, data: dict[str, object]) -> str:
    """Frame one server-sent event.

    `json.dumps` escapes newlines inside strings, so `data` is always a single
    line, which the format requires. The Telegram text has newlines.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_scenario(scenario: str) -> AsyncIterator[str]:
    """Run one scenario through the pipeline, yielding one event per stage.

    Mirrors `process_alert`: no context, no Claude call (invariant 4). Before
    calling Claude it asks whether today's cap has room, because
    `generate_brief` returns None for a spent cap and a failed call alike,
    and invariant 6 must say which happened. `cap_reached` tells the page to
    show its recorded example instead.

    No `try` here. The one broad `except` allowed is `process_alert`'s
    (CLAUDE.md section 6), and everything called here already turns upstream
    failures into None. A bug ends the stream before its `result`; the page
    reports that, and uvicorn logs the traceback.
    """
    yield sse("accepted", {"scenario": scenario})
    # Nothing above this line touched the network. Everything below may.

    if scenario == "malformed":
        payload = demo_alert(MALFORMED_SYMBOL, None)
        alert_id = compute_alert_id(payload)
        context = await get_market_context(payload.symbol, alert_id)
    else:
        started = time.perf_counter()
        contexts, failure = await scan_watchlist()
        symbol = pick_symbol(scenario, contexts) if contexts else WATCHLIST[0]
        context = contexts.get(symbol)
        payload = demo_alert(symbol, context)
        alert_id = compute_alert_id(payload)
        log_event(
            logger,
            alert_id=alert_id,
            stage="demo_scan",
            outcome="ok" if context is not None else "degraded",
            latency_ms=(time.perf_counter() - started) * 1000,
            reason=None if context is not None else failure,
        )

    yield sse(
        "context",
        {
            "alert_id": alert_id,
            "alert": payload.model_dump(mode="json"),
            "context": None if context is None else context.model_dump(),
        },
    )

    brief = None
    outcome = "unenriched"
    if context is not None:
        if daily_cap_allows_a_brief():
            brief = await generate_brief(payload, context, alert_id)
            outcome = "unenriched" if brief is None else "enriched"
        else:
            outcome = "cap_reached"
            log_event(
                logger,
                alert_id=alert_id,
                stage="llm",
                outcome="degraded",
                latency_ms=0.0,
                reason="spend_cap_reached",
            )

    yield sse(
        "result",
        {
            "outcome": outcome,
            "brief": None if brief is None else brief.model_dump(),
            "telegram_text": format_message(payload, brief),
        },
    )


@router.post("/demo/{scenario}")
async def run_demo(scenario: str) -> StreamingResponse:
    """Stream one scenario; 404 when the demo is off or the scenario is unknown.

    Every refusal is decided here, before the stream starts. Once the first
    event is sent the status is fixed at 200, so a later "no" could only
    arrive as a 200 followed by an error.

    A POST, never a GET. Chat apps fetch GET links by themselves to build a
    preview, and every preview would be a Claude call nobody made.
    """
    if not get_settings().demo_enabled or scenario not in SCENARIOS:
        raise HTTPException(status_code=404, detail="Not Found")

    return StreamingResponse(
        stream_scenario(scenario),
        media_type="text/event-stream",
        headers={
            # A cache between us and the visitor must not replay one
            # visitor's result to the next.
            "Cache-Control": "no-cache",
            # The conventional "don't buffer this" for nginx-style proxies.
            # Ignored where it means nothing.
            "X-Accel-Buffering": "no",
        },
    )
