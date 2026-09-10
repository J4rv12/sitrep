"""Binance daily klines -> the two numbers in `MarketContext`.

Invariant 4 at its first stage. If Binance is slow, down, blocking our
region, or has never heard of the symbol, the alert still ships — without
context. So `get_market_context` returns None rather than raising, and says
why in one log line.

Three functions, one direction of travel:

    fetch_klines        all the I/O, and every way it can go wrong
    compute_context     pure arithmetic on floats; no I/O, no clock
    get_market_context  the only entry point the pipeline calls
"""

import logging
import time
from statistics import fmean

import httpx

from app.config import get_settings
from app.logs import log_event
from app.schemas import MarketContext

logger = logging.getLogger(__name__)

KLINES_PATH = "/api/v3/klines"
INTERVAL = "1d"  # Daily whatever the alert's timeframe: the context the alert lacks.
LIMIT = 22  # 20 baseline days + the newest closed day + today's forming bar.

# Positions in a Binance kline row. Prices and volumes arrive as JSON
# strings, not numbers, so no decimal precision is lost in transit.
_CLOSE = 4
_VOLUME = 5


class EnrichmentFailed(Exception):
    """An upstream failure carrying its reason code. Never escapes this module."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def fetch_klines(symbol: str, client: httpx.AsyncClient) -> tuple[list[float], list[float]]:
    """Return `(closes, volumes)` for the last `LIMIT` daily bars, oldest first.

    Both lists are exactly `LIMIT` long. The last element of each is today's
    bar, which is still forming.

    Raises `EnrichmentFailed` with one of these reason codes:

        binance_timeout        no response within the client's timeout
        binance_unreachable    DNS, TLS or connection failure
        binance_status_<code>  any non-200: 400 unknown symbol, 429 rate
                               limited, 451 restricted region, 5xx outage
        binance_malformed      a 200 that is not the JSON array of rows we expect
        binance_short_history  fewer than LIMIT rows: listed under 22 days ago
    """
    params = {"symbol": symbol, "interval": INTERVAL, "limit": LIMIT}
    try:
        response = await client.get(KLINES_PATH, params=params)
    # TimeoutException subclasses TransportError, so it must be caught first.
    except httpx.TimeoutException as e:
        raise EnrichmentFailed("binance_timeout") from e
    except httpx.TransportError as e:
        raise EnrichmentFailed("binance_unreachable") from e

    if response.status_code != 200:
        raise EnrichmentFailed(f"binance_status_{response.status_code}")

    try:
        rows = response.json()
    except ValueError as e:  # json.JSONDecodeError is a ValueError.
        raise EnrichmentFailed("binance_malformed") from e
    if not isinstance(rows, list):
        raise EnrichmentFailed("binance_malformed")
    # Binance never returns more than `limit` rows, so != means fewer.
    if len(rows) != LIMIT:
        raise EnrichmentFailed("binance_short_history")

    try:
        closes = [float(row[_CLOSE]) for row in rows]
        volumes = [float(row[_VOLUME]) for row in rows]
    except (ValueError, TypeError, IndexError, KeyError) as e:
        raise EnrichmentFailed("binance_malformed") from e
    return closes, volumes


def compute_context(closes: list[float], volumes: list[float]) -> MarketContext:
    """Turn 22 daily bars into the two numbers the model will see.

    Takes what `fetch_klines` returns: two lists of exactly 22 floats,
    oldest first, the last one today's still-forming bar. Numbering the
    rows 1 to 22:

        volume_vs_20d_avg = volume of row 21 / mean volume of rows 1-20
        pct_from_20ma     = (close of row 22 - MA) / MA * 100
                            where MA = mean close of rows 3-22

    Why these rows and no others:

    - Row 22's volume is a running total of an unfinished day. It appears
      in no volume calculation — not as the numerator, not in the baseline.
    - Row 21 is the day under test, so it stays out of its own baseline.
      Include it and a 10x day reports as 6.9x.
    - Row 22's close is the latest traded price: a snapshot, and an exact
      one. The 20MA includes it because the 20MA on a TradingView chart
      does, and the alert condition was written against that chart.

    No zero guard on the volume mean. It is zero only after 20 straight
    days without a trade on a pair Binance still serves; of 15 halted pairs
    checked on 2026-09-10, none had more than one such day. If it ever
    happens, ZeroDivisionError reaches Phase 3's last-resort handler and
    the alert ships unenriched.

    Pure — no I/O, no clock, no logging. The same lists give the same
    answer forever, which is what lets the test pin exact values.
    `statistics.fmean` is the mean you want.
    """
    volume = volumes[20] / fmean(volumes[0:20])
    ma = fmean(closes[2:22])
    pct = (closes[21] - ma) / ma * 100

    return MarketContext(volume_vs_20d_avg=volume, pct_from_20ma=pct)


async def get_market_context(
    symbol: str,
    alert_id: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> MarketContext | None:
    """Return market context for one alert, or None if Binance can't supply it.

    Never raises for an upstream failure. Every way Binance can let us down
    becomes None plus one WARNING line carrying a reason code from
    `fetch_klines`'s list, and the alert goes on without context. A bug in
    our own code still raises: catching those here would hide them from the
    tests. `process_alert` gets the last-resort net in Phase 3.

    Logs exactly one line per call, outcome "ok" or "degraded".

    `transport` exists for tests, which pass an `httpx.MockTransport`
    serving recorded responses. Production leaves it None.
    """
    settings = get_settings()
    started = time.perf_counter()
    context: MarketContext | None = None
    reason: str | None = None

    try:
        async with httpx.AsyncClient(
            base_url=settings.binance_base_url,
            timeout=settings.http_timeout_seconds,
            transport=transport,
        ) as client:
            closes, volumes = await fetch_klines(symbol, client)
        context = compute_context(closes, volumes)
    except EnrichmentFailed as e:
        reason = e.reason

    log_event(
        logger,
        alert_id=alert_id,
        stage="enrich",
        outcome="ok" if reason is None else "degraded",
        latency_ms=(time.perf_counter() - started) * 1000,
        reason=reason,
    )
    return context
