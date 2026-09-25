"""The demo endpoint: refusals before the stream, three events in order, and
nothing delivered.

`demo_watchlist_contexts.json` holds market context for the 20 watchlist pairs,
computed by `compute_context` from live Binance responses at
2026-09-24T14:45:59Z. That day only two of the 20 traded below their 20-day
average volume; on the first twelve pairs alone, none did. That is why the
watchlist is twenty pairs long. The brief is the recorded Claude reply
tests/test_llm.py uses, and the klines are the recorded BTCUSDT fixture
tests/test_enrich.py uses. Nothing here is invented.

Most tests replace the four network-facing names in `demo` and read the whole
stream through TestClient. The latency test drives the ASGI app directly, for
the reason tests/test_webhook.py gives: TestClient only returns once the
response has finished.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.routes import alerts, demo
from app.schemas import AlertPayload, MarketContext, SignalBrief
from app.telegram import format_message

FIXTURES = Path(__file__).parent / "fixtures"
RECORDED = json.loads((FIXTURES / "demo_watchlist_contexts.json").read_bytes())
WATCHLIST_CONTEXTS = {s: MarketContext(**c) for s, c in RECORDED["contexts"].items()}
KLINES_BODY = (FIXTURES / "binance_klines_btcusdt_1d.json").read_bytes()
INVALID_SYMBOL_BODY = (FIXTURES / "binance_error_invalid_symbol.json").read_bytes()
_REPLY = json.loads((FIXTURES / "anthropic_brief_btcusdt_low_volume.json").read_bytes())
BRIEF = SignalBrief.model_validate_json(_REPLY["content"][0]["text"])

# The real scan, kept before the `offline` fixture replaces it in every test.
REAL_SCAN = demo.scan_watchlist

client = TestClient(app)


def events(stream: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a server-sent event stream into (event, data) pairs."""
    parsed = []
    for block in stream.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        parsed.append((fields["event"], json.loads(fields["data"])))
    return parsed


def demo_records(caplog: pytest.LogCaptureFixture, stage: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "app.routes.demo" and r.stage == stage]


@pytest.fixture(autouse=True)
def demo_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The demo is on unless a test turns it off.

    Set on the cached settings object, not through the environment, so a
    developer's `.env` saying either value cannot change a result.
    """
    monkeypatch.setattr(get_settings(), "demo_enabled", True)


@pytest.fixture(autouse=True)
def empty_feed() -> None:
    alerts._feed.clear()


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Replace every name that reaches the network, and record what was asked.

    The scan answers with the recorded day. Single-pair enrichment answers
    None, which is what Binance's 400 for `BINANCE:BTCUSDT` becomes. Claude
    answers with the recorded brief. The cap has room.
    """
    calls: dict[str, list[str]] = {"scan": [], "enrich": [], "brief": []}

    async def recorded_day() -> tuple[dict[str, MarketContext], str | None]:
        calls["scan"].append("watchlist")
        return WATCHLIST_CONTEXTS, None

    async def no_context(symbol: str, alert_id: str) -> None:
        calls["enrich"].append(symbol)
        return None

    async def recorded_brief(
        payload: AlertPayload, context: MarketContext, alert_id: str
    ) -> SignalBrief:
        calls["brief"].append(payload.symbol)
        return BRIEF

    monkeypatch.setattr(demo, "scan_watchlist", recorded_day)
    monkeypatch.setattr(demo, "get_market_context", no_context)
    monkeypatch.setattr(demo, "generate_brief", recorded_brief)
    monkeypatch.setattr(demo, "daily_cap_allows_a_brief", lambda: True)
    return calls


# --- refusals, all before the stream starts ---------------------------------


def test_demo_is_404_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "demo_enabled", False)

    assert client.post("/demo/breakout").status_code == 404


def test_unknown_scenario_is_404() -> None:
    assert client.post("/demo/moon").status_code == 404


def test_get_does_not_run_a_scenario(offline: dict[str, list[str]]) -> None:
    """A GET is what a chat app's link preview sends."""
    assert client.get("/demo/breakout").status_code == 405
    assert offline["scan"] == []


# --- the stream -------------------------------------------------------------


def test_stream_is_three_events_in_order() -> None:
    response = client.post("/demo/breakout")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _ in events(response.text)] == ["accepted", "context", "result"]


def test_breakout_runs_the_picked_pair_through_claude(offline: dict[str, list[str]]) -> None:
    (_, accepted), (_, context), (_, result) = events(client.post("/demo/breakout").text)

    assert accepted == {"scenario": "breakout"}
    assert context["alert"]["symbol"] == "LINKUSDT"
    assert context["context"] == WATCHLIST_CONTEXTS["LINKUSDT"].model_dump()
    assert offline["brief"] == ["LINKUSDT"]
    assert result["outcome"] == "enriched"
    assert result["brief"] == BRIEF.model_dump()


def test_result_carries_the_exact_telegram_text() -> None:
    _, (_, context), (_, result) = events(client.post("/demo/weak").text)

    payload = AlertPayload.model_validate(context["alert"])
    assert result["telegram_text"] == format_message(payload, BRIEF)


# --- picking a pair: live data, so tested on the recorded day ---------------


def test_breakout_is_the_strongest_volume_within_5pct_of_the_20ma() -> None:
    """BCH had 4.5x volume at +32.9% and DOGE 2.0x at +8.6%: strong, but
    stretched. LINK, 1.6x at +4.7%, was the clean one."""
    assert demo.pick_symbol("breakout", WATCHLIST_CONTEXTS) == "LINKUSDT"


def test_breakout_falls_back_to_every_pair_when_all_are_stretched() -> None:
    stretched = {s: WATCHLIST_CONTEXTS[s] for s in ("BCHUSDT", "NEARUSDT", "LTCUSDT")}

    assert demo.pick_symbol("breakout", stretched) == "BCHUSDT"


def test_weak_is_the_lowest_volume() -> None:
    """TON, at 0.63x, was one of only two pairs below average that day."""
    assert demo.pick_symbol("weak", WATCHLIST_CONTEXTS) == "TONUSDT"


def test_condition_names_the_side_of_the_20ma_price_is_on() -> None:
    below = demo.demo_alert("TONUSDT", WATCHLIST_CONTEXTS["TONUSDT"])  # -2.4%
    above = demo.demo_alert("LINKUSDT", WATCHLIST_CONTEXTS["LINKUSDT"])  # +4.7%

    assert (below.condition, above.condition) == ("close below 20MA", "close above 20MA")


# --- degrading inside the stream (invariant 4) ------------------------------


def test_malformed_symbol_degrades_without_calling_claude(offline: dict[str, list[str]]) -> None:
    _, (_, context), (_, result) = events(client.post("/demo/malformed").text)

    assert offline["enrich"] == ["BINANCE:BTCUSDT"]
    assert offline["brief"] == [], "no context must mean no Claude call"
    assert context["context"] is None
    assert (result["outcome"], result["brief"]) == ("unenriched", None)
    assert result["telegram_text"].startswith("[unenriched]")


def test_no_pair_answering_degrades_without_claude(
    monkeypatch: pytest.MonkeyPatch,
    offline: dict[str, list[str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def binance_down() -> tuple[dict[str, MarketContext], str | None]:
        return {}, "binance_timeout"

    monkeypatch.setattr(demo, "scan_watchlist", binance_down)

    _, (_, context), (_, result) = events(client.post("/demo/breakout").text)

    assert context["context"] is None
    assert offline["brief"] == []
    assert result["outcome"] == "unenriched"
    (record,) = demo_records(caplog, "demo_scan")
    assert (record.outcome, record.reason) == ("degraded", "binance_timeout")


def test_failed_brief_is_unenriched(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_brief(payload: AlertPayload, context: MarketContext, alert_id: str) -> None:
        return None

    monkeypatch.setattr(demo, "generate_brief", no_brief)

    _, (_, context), (_, result) = events(client.post("/demo/breakout").text)

    assert context["context"] is not None
    assert (result["outcome"], result["brief"]) == ("unenriched", None)


# --- the cap (invariant 6) --------------------------------------------------


def test_spent_cap_skips_claude_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    offline: dict[str, list[str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The page shows its recorded example on `cap_reached`, so the outcome
    must name the cap, not just report a missing brief."""
    monkeypatch.setattr(demo, "daily_cap_allows_a_brief", lambda: False)

    _, _, (_, result) = events(client.post("/demo/weak").text)

    assert offline["brief"] == []
    assert result["outcome"] == "cap_reached"
    (record,) = demo_records(caplog, "llm")
    assert (record.outcome, record.reason) == ("degraded", "spend_cap_reached")


# --- nothing leaves the demo ------------------------------------------------


def test_demo_delivers_nothing_and_records_nothing() -> None:
    """A visitor's click must not message your chat, and must not push your
    real alerts out of the feed's 50 entries."""
    for scenario in demo.SCENARIOS:
        client.post(f"/demo/{scenario}")

    assert len(alerts._feed) == 0
    assert not hasattr(demo, "send_message"), "demo.py must not be able to reach Telegram"


# --- scan_watchlist: the demo's own Binance I/O -----------------------------


async def test_scan_keeps_the_pairs_that_answer_and_names_the_first_failure() -> None:
    """BTCUSDT gets the recorded klines; every other pair the recorded 400."""
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["symbol"]
        asked.append(symbol)
        if symbol == "BTCUSDT":
            return httpx.Response(200, content=KLINES_BODY)
        return httpx.Response(400, content=INVALID_SYMBOL_BODY)

    contexts, failure = await REAL_SCAN(transport=httpx.MockTransport(handler))

    assert sorted(asked) == sorted(demo.WATCHLIST)
    assert list(contexts) == ["BTCUSDT"]
    assert contexts["BTCUSDT"].volume_vs_20d_avg == pytest.approx(0.7739972486180567, rel=1e-9)
    assert failure == "binance_status_400"


async def test_scan_with_every_pair_failing_returns_nothing_and_why() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated", request=request)

    assert await REAL_SCAN(transport=httpx.MockTransport(handler)) == ({}, "binance_timeout")


# --- invariant 1, on the demo -----------------------------------------------


async def post_measuring_first_event(path: str) -> tuple[bytes, float, float]:
    """POST `path`, driving the ASGI app directly.

    Returns (first body chunk, ms until it was sent, ms until the app
    returned). `receive` answers once and then waits forever, like a client
    that sent its request and is still listening.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    requested = False
    seen: dict[str, Any] = {}

    async def receive() -> dict[str, Any]:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body") and "first" not in seen:
            seen["first"] = message["body"]
            seen["first_at"] = time.perf_counter()

    start = time.perf_counter()
    await app(scope, receive, send)
    finished = time.perf_counter()

    return seen["first"], (seen["first_at"] - start) * 1000, (finished - start) * 1000


async def test_accepted_is_sent_before_binance_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scan is replaced with a one-second sleep. `accepted` must be on the
    wire long before it ends, and the last assertion proves the scan did run."""

    async def slow_scan() -> tuple[dict[str, MarketContext], str | None]:
        await asyncio.sleep(1)
        return WATCHLIST_CONTEXTS, None

    monkeypatch.setattr(demo, "scan_watchlist", slow_scan)

    first, first_ms, finished_ms = await post_measuring_first_event("/demo/breakout")

    assert first.startswith(b"event: accepted")
    assert first_ms < 500, f"first event took {first_ms:.1f} ms to send"
    assert finished_ms > 900, "the slow scan did not actually run"
