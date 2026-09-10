"""Enrichment: the two numbers, and every way Binance can fail to supply them.

Fixtures are real, recorded byte for byte from Binance's public market-data
host at 2026-09-10T11:21:25Z:

    binance_klines_btcusdt_1d.json     /api/v3/klines?symbol=BTCUSDT&interval=1d&limit=22
    binance_error_invalid_symbol.json  the same with symbol=NOTASYMBOL (HTTP 400)

Row 22 was eleven hours into its day when recorded: 5,712 BTC traded against
row 21's full-day 14,130. The partial volume `compute_context` must ignore is
in the data, not described.

Failures a fixture cannot hold — a timeout, a refused connection, a 500 —
are simulated at the transport, the only place they can honestly be made.
No price or volume in this file is invented.
"""

import json
import logging
from pathlib import Path

import httpx
import pytest

from app.enrich import LIMIT, compute_context, get_market_context

FIXTURES = Path(__file__).parent / "fixtures"
KLINES_BODY = (FIXTURES / "binance_klines_btcusdt_1d.json").read_bytes()
INVALID_SYMBOL_BODY = (FIXTURES / "binance_error_invalid_symbol.json").read_bytes()

ROWS = json.loads(KLINES_BODY)
CLOSES = [float(row[4]) for row in ROWS]
VOLUMES = [float(row[5]) for row in ROWS]


@pytest.fixture(autouse=True)
def capture_info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.enrich")


def enrich_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "app.enrich"]


# --- compute_context: yours -------------------------------------------------

# Pinned from the fixture. Every other value below is one specific mistake,
# computed from the same data, so a failing assertion names it. A mistake
# not on the list still reports how far off it is, which is often enough.
EXPECTED_VOLUME_RATIO = 0.7739972486180567
EXPECTED_PCT_FROM_MA = -0.8850766555007303

VOLUME_MISTAKES = {
    0.8230846251581752: "baseline is rows 2-21, so row 21 is inside its own baseline",
    0.7824176554206702: "baseline is rows 1-21, so row 21 is inside its own baseline",
    0.3129129148326476: "numerator is row 22's partial volume",
    0.33275804232125433: "numerator is row 22's partial volume, baseline rows 2-21",
}
PCT_MISTAKES = {
    -0.9070814941623382: "MA is rows 2-21; it must end at row 22, like the chart's",
    -0.5734837673865654: "MA is rows 1-20; it must end at row 22, like the chart's",
    -0.5032059637318638: "price is row 21's close, a day old; use row 22's",
    -0.8929802149212385: "divided by the price; divide by the MA",
    -0.008850766555007303: "a fraction, not a percentage",
}


def diagnose(actual: float, expected: float, mistakes: dict[float, str]) -> str:
    for value, mistake in mistakes.items():
        if actual == pytest.approx(value, rel=1e-9):
            return f"got {actual}: {mistake}"
    if actual == 0:
        return f"got 0, expected {expected}: not one of the known mistakes"
    return (
        f"got {actual}, expected {expected}, a factor of {expected / actual:.6g} "
        "apart: not one of the known mistakes"
    )


def test_volume_ratio_is_newest_closed_day_against_the_20_before_it() -> None:
    ratio = compute_context(CLOSES, VOLUMES).volume_vs_20d_avg

    expected = EXPECTED_VOLUME_RATIO
    assert ratio == pytest.approx(expected, rel=1e-9), diagnose(ratio, expected, VOLUME_MISTAKES)


def test_pct_from_20ma_is_live_price_against_a_20ma_ending_today() -> None:
    pct = compute_context(CLOSES, VOLUMES).pct_from_20ma

    expected = EXPECTED_PCT_FROM_MA
    assert pct == pytest.approx(expected, rel=1e-9), diagnose(pct, expected, PCT_MISTAKES)


# --- get_market_context: the pipeline's entry point ------------------------


def serve(
    status: int, body: bytes = b"", seen: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    """A transport that answers every request with `status` and `body`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, content=body)

    return httpx.MockTransport(handler)


def fail_with(error: type[httpx.TransportError]) -> httpx.MockTransport:
    """A transport that raises the way httpx does when the network fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error("simulated", request=request)

    return httpx.MockTransport(handler)


async def test_recorded_response_becomes_market_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[httpx.Request] = []

    context = await get_market_context(
        "BTCUSDT", "alert-1", transport=serve(200, KLINES_BODY, seen)
    )

    assert context is not None
    assert context.volume_vs_20d_avg == pytest.approx(EXPECTED_VOLUME_RATIO, rel=1e-9)
    assert context.pct_from_20ma == pytest.approx(EXPECTED_PCT_FROM_MA, rel=1e-9)

    (request,) = seen
    assert request.url.path == "/api/v3/klines"
    assert dict(request.url.params) == {"symbol": "BTCUSDT", "interval": "1d", "limit": "22"}

    (record,) = enrich_records(caplog)
    assert (record.levelname, record.stage, record.outcome) == ("INFO", "enrich", "ok")


# Not one of these may raise. An exception here would escape a background
# task after the 202 was sent, and the alert would be dropped (Phase 1, Q3).
UPSTREAM_FAILURES = [
    pytest.param(fail_with(httpx.ReadTimeout), "binance_timeout", id="read-timeout"),
    pytest.param(fail_with(httpx.ConnectTimeout), "binance_timeout", id="connect-timeout"),
    pytest.param(fail_with(httpx.ConnectError), "binance_unreachable", id="unreachable"),
    pytest.param(serve(500), "binance_status_500", id="upstream-500"),
    pytest.param(serve(400, INVALID_SYMBOL_BODY), "binance_status_400", id="unknown-symbol"),
    pytest.param(serve(200, b"<html></html>"), "binance_malformed", id="not-json"),
    pytest.param(serve(200, INVALID_SYMBOL_BODY), "binance_malformed", id="object-not-array"),
    pytest.param(serve(200, json.dumps([{}] * LIMIT).encode()), "binance_malformed", id="bad-rows"),
    pytest.param(
        serve(200, json.dumps(ROWS[:5]).encode()), "binance_short_history", id="short-history"
    ),
]


@pytest.mark.parametrize(("transport", "reason"), UPSTREAM_FAILURES)
async def test_upstream_failure_degrades_to_none_with_a_reason_code(
    transport: httpx.MockTransport, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    context = await get_market_context("BTCUSDT", "alert-1", transport=transport)

    assert context is None
    (record,) = enrich_records(caplog)
    assert record.levelname == "WARNING"
    assert (record.alert_id, record.stage, record.outcome, record.reason) == (
        "alert-1",
        "enrich",
        "degraded",
        reason,
    )
