"""The feed: private, newest first, and bounded.

The brief is the recorded Claude reply tests/test_llm.py uses. Entries are
recorded directly with `record_alert`; the pipeline recording them after
delivery is tested in tests/test_webhook.py.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes import alerts
from app.schemas import AlertPayload, FeedEntry, SignalBrief

TOKEN = "test-webhook-token"
HEADERS = {"X-SitRep-Token": TOKEN}

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

client = TestClient(app)


@pytest.fixture(autouse=True)
def empty_feed() -> None:
    """Start every test with the real feed emptied.

    Emptied, not replaced. A replacement built here would carry its own
    maxlen, and a module that shipped `deque()` with no bound would pass
    every test below while leaking memory in production.
    """
    alerts._feed.clear()


def entry(alert_id: str, brief: SignalBrief | None = None, delivered: bool = True) -> FeedEntry:
    return FeedEntry(
        alert_id=alert_id,
        alert=ALERT,
        brief=brief,
        delivered=delivered,
        recorded_at=datetime.fromisoformat("2026-09-10T10:00:05+00:00"),
    )


def feed_ids() -> list[str]:
    return [item["alert_id"] for item in client.get("/alerts", headers=HEADERS).json()]


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-SitRep-Token": "not-the-token"}, {"Authorization": f"Bearer {TOKEN}"}],
    ids=["missing", "wrong", "wrong-header"],
)
def test_feed_requires_the_token_header(headers: dict[str, str]) -> None:
    alerts.record_alert(entry("alert-1"))

    response = client.get("/alerts", headers=headers)

    assert response.status_code == 401
    assert "alert-1" not in response.text


def test_token_in_the_query_string_is_not_accepted() -> None:
    """A query string lands in Render's access log. Accepting it invites using it."""
    assert client.get("/alerts", params={"token": TOKEN}).status_code == 401


def test_empty_feed_is_an_empty_list() -> None:
    response = client.get("/alerts", headers=HEADERS)

    assert response.status_code == 200
    assert response.json() == []


def test_entry_shape() -> None:
    """What a reader of the feed gets, asserted whole. Both outcomes, newest first."""
    alerts.record_alert(entry("alert-1", brief=None, delivered=False))
    alerts.record_alert(entry("alert-2", brief=BRIEF, delivered=True))

    alert_json = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "condition": "close below 20MA",
        "bar_time": "2026-09-10T10:00:00Z",
        "price": None,
    }
    assert client.get("/alerts", headers=HEADERS).json() == [
        {
            "alert_id": "alert-2",
            "alert": alert_json,
            "brief": BRIEF.model_dump(),
            "delivered": True,
            "recorded_at": "2026-09-10T10:00:05Z",
        },
        {
            "alert_id": "alert-1",
            "alert": alert_json,
            "brief": None,
            "delivered": False,
            "recorded_at": "2026-09-10T10:00:05Z",
        },
    ]


def test_feed_is_newest_first() -> None:
    for alert_id in ["first", "second", "third"]:
        alerts.record_alert(entry(alert_id))

    assert feed_ids() == ["third", "second", "first"]


def test_feed_keeps_only_the_newest_entries() -> None:
    """The memory bound. One past the limit drops exactly the oldest."""
    for n in range(alerts.FEED_SIZE + 1):
        alerts.record_alert(entry(f"alert-{n}"))

    ids = feed_ids()

    assert len(ids) == alerts.FEED_SIZE
    assert ids[0] == f"alert-{alerts.FEED_SIZE}"
    assert "alert-0" not in ids


def test_a_new_entry_gets_the_time_it_was_recorded() -> None:
    """The pipeline passes no timestamp; the model stamps one, in UTC."""
    before = datetime.now(UTC)

    stamped = FeedEntry(alert_id="alert-1", alert=ALERT, brief=None, delivered=True)

    assert stamped.recorded_at.tzinfo is UTC
    assert stamped.recorded_at >= before
