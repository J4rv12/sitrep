"""The webhook contract, including the latency assertion that is the point
of Phase 1.

Two ways of calling the app here, and the difference matters.

`TestClient` is used for everything about status codes and bodies. It is
convenient and it runs the whole ASGI cycle — *including background tasks* —
before `post()` returns. That makes it useless for timing: a correct handler
that enqueues a five-second task takes five seconds to come back through
TestClient. Measured, not assumed:

    TestClient.post() returned after 1016.8 ms   (task slept 1.0 s)

So `send_webhook_measuring_send` drives the ASGI app directly and timestamps
the moment the response body is handed to the transport. That is the number
invariant 1 is actually about — when TradingView stops waiting — and it
separates cleanly from the task:

    response sent after 0.4 ms | app finished after 1001.7 ms
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dedupe import DedupeCache, compute_alert_id
from app.enrich import compute_context
from app.main import app
from app.routes import alerts, webhook
from app.schemas import AlertPayload, MarketContext, SignalBrief
from app.telegram import format_message
from conftest import FakeClock

TOKEN = "test-webhook-token-at-least-32-chars"

ALERT = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "condition": "close above 20MA",
    "bar_time": "2026-09-04T10:00:00Z",
    "price": 64000.0,
}

client = TestClient(app)


def body(**overrides: Any) -> dict[str, Any]:
    """A well-formed request body: the alert plus a valid token."""
    return {**ALERT, "token": TOKEN} | overrides


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Give every test its own empty cache on a clock it controls.

    `_dedupe` is module-level state built at import, so without this the
    second test to send ALERT would see a duplicate left by the first, and
    the suite would pass or fail depending on test order.

    Reaching into a private name is acceptable here because this file tests
    that module. Another *production* module doing it would be the coupling
    the underscore exists to warn about.
    """
    clock = FakeClock()
    monkeypatch.setattr(webhook, "_dedupe", DedupeCache(ttl_seconds=300, clock=clock))
    return clock


@pytest.fixture(autouse=True)
def offline_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test off the network.

    `TestClient` runs background tasks before `post()` returns, so every
    accepted alert in this file runs the whole pipeline. Enrichment is
    replaced with "no context", which ends `process_alert` before Claude;
    `generate_brief` is replaced with a failure, so a test that reaches it by
    accident says so; `send_message` reports success without sending.

    `raising=False` because `send_message` exists in `webhook` only once
    `deliver_alert` imports it; until then that patch is harmless.
    """

    async def no_context(symbol: str, alert_id: str) -> None:
        return None

    async def must_not_be_called(*args: object, **kwargs: object) -> None:
        pytest.fail("generate_brief was called; this test is not offline")

    async def delivered(text: str, alert_id: str) -> bool:
        return True

    monkeypatch.setattr(webhook, "get_market_context", no_context)
    monkeypatch.setattr(webhook, "generate_brief", must_not_be_called)
    monkeypatch.setattr(webhook, "send_message", delivered, raising=False)


@pytest.fixture(autouse=True)
def empty_feed() -> None:
    """Accepted alerts land in the feed, so start every test with it empty."""
    alerts._feed.clear()


async def send_webhook_measuring_send(payload: bytes) -> tuple[int, float, float]:
    """POST `payload` to /webhook, driving the ASGI app directly.

    Returns (status, ms until the response was sent, ms until the app
    returned). The gap between the two is the background task.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/webhook",
        "raw_path": b"/webhook",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    seen: dict[str, Any] = {}

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            seen["status"] = message["status"]
        elif message["type"] == "http.response.body":
            seen.setdefault("sent_at", time.perf_counter())

    start = time.perf_counter()
    await app(scope, receive, send)
    finished = time.perf_counter()

    return seen["status"], (seen["sent_at"] - start) * 1000, (finished - start) * 1000


# --- the happy path ---------------------------------------------------------


def test_valid_alert_is_accepted() -> None:
    response = client.post("/webhook", json=body())

    assert response.status_code == 202
    assert response.json()["status"] == "accepted"


def test_response_carries_the_alert_id() -> None:
    """The id is the caller's receipt and the key in every later log line."""
    expected = compute_alert_id(AlertPayload(**ALERT))

    response = client.post("/webhook", json=body())

    assert response.json()["alert_id"] == expected


def test_accepted_alert_is_enqueued() -> None:
    calls: list[tuple[AlertPayload, str]] = []

    async def spy(payload: AlertPayload, alert_id: str) -> None:
        calls.append((payload, alert_id))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(webhook, "deliver_alert", spy)
        response = client.post("/webhook", json=body())

    assert len(calls) == 1
    assert calls[0][0].symbol == "BTCUSDT"
    assert calls[0][1] == response.json()["alert_id"]


# --- authentication ---------------------------------------------------------


def test_missing_token_is_rejected() -> None:
    payload = body()
    del payload["token"]

    assert client.post("/webhook", json=payload).status_code == 401


def test_wrong_token_is_rejected() -> None:
    assert client.post("/webhook", json=body(token="not-the-token")).status_code == 401


def test_authentication_precedes_validation() -> None:
    """A wrong token with a broken alert is a 401, never a 422.

    Otherwise the endpoint is a free schema oracle: send garbage with any
    token and the validation errors tell you every field name we expect.
    """
    response = client.post("/webhook", json={"token": "wrong", "symbol": ""})

    assert response.status_code == 401


def test_rejected_alert_is_not_enqueued() -> None:
    calls: list[str] = []

    async def spy(payload: AlertPayload, alert_id: str) -> None:
        calls.append(alert_id)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(webhook, "deliver_alert", spy)
        client.post("/webhook", json=body(token="not-the-token"))

    assert calls == []


# --- malformed input --------------------------------------------------------


def test_malformed_json_is_rejected() -> None:
    response = client.post(
        "/webhook",
        content=b'{"symbol": "BTCUSDT",',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400


@pytest.mark.parametrize("payload", [b"[1, 2, 3]", b'"just a string"', b"null"])
def test_json_that_is_not_an_object_is_rejected(payload: bytes) -> None:
    """Valid JSON, but there is nowhere for a token to live."""
    response = client.post(
        "/webhook",
        content=payload,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", ""),
        ("timeframe", ""),
        ("condition", ""),
        ("bar_time", "not-a-timestamp"),
    ],
)
def test_invalid_alert_fields_are_rejected(field: str, value: str) -> None:
    response = client.post("/webhook", json=body(**{field: value}))

    assert response.status_code == 422


def test_missing_required_field_is_rejected() -> None:
    payload = body()
    del payload["symbol"]

    assert client.post("/webhook", json=payload).status_code == 422


# --- idempotency ------------------------------------------------------------


def test_duplicate_within_ttl_is_reported_not_reprocessed() -> None:
    calls: list[str] = []

    async def spy(payload: AlertPayload, alert_id: str) -> None:
        calls.append(alert_id)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(webhook, "deliver_alert", spy)
        first = client.post("/webhook", json=body())
        second = client.post("/webhook", json=body())

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert second.json()["alert_id"] == first.json()["alert_id"]
    assert len(calls) == 1, "the duplicate must not reach the pipeline"


def test_duplicate_after_ttl_expiry_is_accepted_again(fresh_cache: FakeClock) -> None:
    """Same alert, six minutes later. The window has closed; process it."""
    assert client.post("/webhook", json=body()).status_code == 202

    fresh_cache.advance(301)

    assert client.post("/webhook", json=body()).status_code == 202


def test_price_change_does_not_defeat_dedupe() -> None:
    """A retry can carry a different price and is still the same alert."""
    client.post("/webhook", json=body(price=64000.0))

    assert client.post("/webhook", json=body(price=64100.0)).status_code == 200


# --- invariant 1 ------------------------------------------------------------


def test_handler_returns_before_the_pipeline_runs() -> None:
    """The phase, in one assertion.

    `deliver_alert` is replaced with a five-second sleep — a stand-in for
    Binance plus Claude plus Telegram, which really do take seconds. The
    response must still be on the wire in well under 500 ms.

    The second assertion is not decoration. It proves the slow task actually
    ran, so a handler that passed by silently dropping the work would fail
    here rather than look fast.
    """

    async def slow_pipeline(payload: AlertPayload, alert_id: str) -> None:
        await asyncio.sleep(5)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(webhook, "deliver_alert", slow_pipeline)
        status, sent_ms, finished_ms = asyncio.run(
            send_webhook_measuring_send(json.dumps(body()).encode())
        )

    assert status == 202
    assert sent_ms < 500, f"response took {sent_ms:.1f} ms to send"
    assert finished_ms > 4900, "the background task did not actually run"


# --- process_alert: yours ---------------------------------------------------

# The context the recorded BTCUSDT klines produce, as in tests/test_llm.py.
_KLINES = json.loads(
    (Path(__file__).parent / "fixtures" / "binance_klines_btcusdt_1d.json").read_bytes()
)
CONTEXT = compute_context([float(r[4]) for r in _KLINES], [float(r[5]) for r in _KLINES])
PAYLOAD = AlertPayload(**ALERT)
BRIEF = SignalBrief(
    severity="low",
    headline="BTCUSDT close above 20MA on below-average volume",
    observations=["Volume is 0.8x its 20-day average.", "Price is 0.9% below its 20-day average."],
)


@pytest.fixture(autouse=True)
def capture_pipeline_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.routes.webhook")


def pipeline_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "app.routes.webhook"]


def enrichment_returning(context: MarketContext | None, seen: list[tuple[str, str]]) -> Any:
    async def fake(symbol: str, alert_id: str) -> MarketContext | None:
        seen.append((symbol, alert_id))
        return context

    return fake


def brief_returning(brief: SignalBrief | None, seen: list[tuple[Any, ...]]) -> Any:
    async def fake(
        payload: AlertPayload, context: MarketContext, alert_id: str
    ) -> SignalBrief | None:
        seen.append((payload, context, alert_id))
        return brief

    return fake


def raising(error: BaseException) -> Any:
    async def fake(*args: object, **kwargs: object) -> None:
        raise error

    return fake


async def test_context_and_brief_flow_through(monkeypatch: pytest.MonkeyPatch) -> None:
    enrich_calls: list[tuple[str, str]] = []
    brief_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(webhook, "get_market_context", enrichment_returning(CONTEXT, enrich_calls))
    monkeypatch.setattr(webhook, "generate_brief", brief_returning(BRIEF, brief_calls))

    result = await webhook.process_alert(PAYLOAD, "alert-1")

    assert result == BRIEF
    assert enrich_calls == [("BTCUSDT", "alert-1")]
    assert brief_calls == [(PAYLOAD, CONTEXT, "alert-1")]


async def test_no_context_means_claude_is_not_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 4: nothing to report, so nothing to pay for."""
    enrich_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(webhook, "get_market_context", enrichment_returning(None, enrich_calls))
    # generate_brief stays as offline_pipeline left it: calling it fails the test.

    result = await webhook.process_alert(PAYLOAD, "alert-1")

    assert result is None
    assert enrich_calls == [("BTCUSDT", "alert-1")]


async def test_a_brief_that_degraded_to_none_is_passed_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhook, "get_market_context", enrichment_returning(CONTEXT, []))
    monkeypatch.setattr(webhook, "generate_brief", brief_returning(None, []))

    assert await webhook.process_alert(PAYLOAD, "alert-1") is None


@pytest.mark.parametrize("stage", ["get_market_context", "generate_brief"])
async def test_a_bug_in_the_pipeline_is_logged_not_raised(
    stage: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The last-resort net. Without it the alert is dropped and nothing says so."""
    monkeypatch.setattr(webhook, "get_market_context", enrichment_returning(CONTEXT, []))
    monkeypatch.setattr(webhook, stage, raising(ZeroDivisionError("simulated")))

    result = await webhook.process_alert(PAYLOAD, "alert-1")

    assert result is None
    (record,) = pipeline_records(caplog)
    assert (record.levelname, record.alert_id, record.stage, record.outcome, record.reason) == (
        "ERROR",
        "alert-1",
        "pipeline",
        "degraded",
        "internal_error",
    )
    assert record.exc_info is not None and record.exc_info[0] is ZeroDivisionError


async def test_cancellation_passes_through_the_net(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown cancels running tasks. A net that swallows it leaves zombies."""
    monkeypatch.setattr(webhook, "get_market_context", raising(asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        await webhook.process_alert(PAYLOAD, "alert-1")


async def test_expected_degradation_leaves_no_pipeline_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Enrichment and the LLM log their own outcomes; a second line would double-count."""
    monkeypatch.setattr(webhook, "get_market_context", enrichment_returning(None, []))

    await webhook.process_alert(PAYLOAD, "alert-1")

    assert pipeline_records(caplog) == []


# --- deliver_alert: yours ---------------------------------------------------


def pipeline_returning(brief: SignalBrief | None, seen: list[tuple[Any, ...]]) -> Any:
    async def fake(payload: AlertPayload, alert_id: str) -> SignalBrief | None:
        seen.append((payload, alert_id))
        return brief

    return fake


def telegram_answering(delivered: bool, sent: list[tuple[str, str]]) -> Any:
    async def fake(text: str, alert_id: str) -> bool:
        sent.append((text, alert_id))
        return delivered

    return fake


async def test_a_brief_is_sent_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_calls: list[tuple[Any, ...]] = []
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(webhook, "process_alert", pipeline_returning(BRIEF, pipeline_calls))
    monkeypatch.setattr(webhook, "send_message", telegram_answering(True, sent), raising=False)

    await webhook.deliver_alert(PAYLOAD, "alert-1")

    assert pipeline_calls == [(PAYLOAD, "alert-1")]
    assert sent == [(format_message(PAYLOAD, BRIEF), "alert-1")]
    (entry,) = alerts._feed
    assert (entry.alert_id, entry.alert, entry.brief, entry.delivered) == (
        "alert-1",
        PAYLOAD,
        BRIEF,
        True,
    )


async def test_no_brief_still_ships_tagged_unenriched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 4: degrade, never drop."""
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(webhook, "process_alert", pipeline_returning(None, []))
    monkeypatch.setattr(webhook, "send_message", telegram_answering(True, sent), raising=False)

    await webhook.deliver_alert(PAYLOAD, "alert-1")

    assert sent == [(format_message(PAYLOAD, None), "alert-1")]
    (entry,) = alerts._feed
    assert (entry.brief, entry.delivered) == (None, True)


async def test_a_bug_caught_by_the_net_still_ships(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why delivery sits after the net, not inside it.

    The real `process_alert` runs here. Enrichment raises, the net turns it
    into None, and the alert must still reach the chat, unenriched.
    """
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(webhook, "get_market_context", raising(ZeroDivisionError("simulated")))
    monkeypatch.setattr(webhook, "send_message", telegram_answering(True, sent), raising=False)

    await webhook.deliver_alert(PAYLOAD, "alert-1")

    assert sent == [(format_message(PAYLOAD, None), "alert-1")]


async def test_a_refused_send_is_recorded_as_undelivered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The feed says what happened; Telegram's reason code is in its own log line."""
    monkeypatch.setattr(webhook, "process_alert", pipeline_returning(BRIEF, []))
    monkeypatch.setattr(webhook, "send_message", telegram_answering(False, []), raising=False)

    await webhook.deliver_alert(PAYLOAD, "alert-1")

    (entry,) = alerts._feed
    assert (entry.brief, entry.delivered) == (BRIEF, False)


def test_accepted_alert_reaches_the_feed_without_its_token() -> None:
    """The whole path through the app: webhook in, feed out.

    The token arrives in the body, and nothing downstream may keep it.
    """
    accepted = client.post("/webhook", json=body())

    listed = client.get("/alerts", headers={"X-SitRep-Token": TOKEN})

    (item,) = listed.json()
    assert item["alert_id"] == accepted.json()["alert_id"]
    assert item["brief"] is None, "offline_pipeline supplies no context"
    assert TOKEN not in listed.text
