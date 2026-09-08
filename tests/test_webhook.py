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
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dedupe import DedupeCache, compute_alert_id
from app.main import app
from app.routes import webhook
from app.schemas import AlertPayload
from conftest import FakeClock

TOKEN = "test-webhook-token"

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
        patch.setattr(webhook, "process_alert", spy)
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
        patch.setattr(webhook, "process_alert", spy)
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
        patch.setattr(webhook, "process_alert", spy)
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

    `process_alert` is replaced with a five-second sleep — a stand-in for
    Binance plus Claude plus Telegram, which really do take seconds. The
    response must still be on the wire in well under 500 ms.

    The second assertion is not decoration. It proves the slow task actually
    ran, so a handler that passed by silently dropping the work would fail
    here rather than look fast.
    """

    async def slow_pipeline(payload: AlertPayload, alert_id: str) -> None:
        await asyncio.sleep(5)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(webhook, "process_alert", slow_pipeline)
        status, sent_ms, finished_ms = asyncio.run(
            send_webhook_measuring_send(json.dumps(body()).encode())
        )

    assert status == 202
    assert sent_ms < 500, f"response took {sent_ms:.1f} ms to send"
    assert finished_ms > 4900, "the background task did not actually run"
