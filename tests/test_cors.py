"""CORS: which web pages a browser lets read this service's responses.

A browser sends `Origin` with a cross-site request and only hands the response
to the page if `Access-Control-Allow-Origin` names that origin. These tests
play the browser's half by sending `Origin` themselves.

The allowed origin is read from settings rather than written out, so a
developer's `.env` adding a local origin cannot make these fail.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.routes import demo
from app.security import RateLimiter
from conftest import FakeClock

PAGE = get_settings().demo_allowed_origins[0]
ELSEWHERE = "https://some-other-site.example"

client = TestClient(app)


@pytest.fixture(autouse=True)
def demo_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The demo on, its limiter fresh, and single-pair enrichment off the network."""

    async def no_context(symbol: str, alert_id: str) -> None:
        return None

    monkeypatch.setattr(get_settings(), "demo_enabled", True)
    monkeypatch.setattr(demo, "get_market_context", no_context)
    limiter = RateLimiter(limit=10, window_seconds=3600, clock=FakeClock())
    monkeypatch.setattr(demo, "_limiter", limiter)


def test_the_demo_page_may_read_the_warm_up_ping() -> None:
    response = client.get("/healthz", headers={"Origin": PAGE})

    assert response.headers["access-control-allow-origin"] == PAGE


def test_the_demo_page_may_read_the_demo_stream() -> None:
    response = client.post("/demo/malformed", headers={"Origin": PAGE})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == PAGE


def test_another_site_may_not_read_responses() -> None:
    response = client.post("/demo/malformed", headers={"Origin": ELSEWHERE})

    assert "access-control-allow-origin" not in response.headers


def test_the_page_cannot_be_used_to_read_the_feed() -> None:
    """Reading the feed needs the `X-SitRep-Token` header, so a browser asks
    first with a preflight. No headers are allowed, so the answer is no."""
    preflight = client.options(
        "/alerts",
        headers={
            "Origin": PAGE,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-sitrep-token",
        },
    )

    assert preflight.status_code == 400
