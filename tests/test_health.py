"""Health endpoint contract.

Small, but it proves the whole scaffold: config validated at import, the app
object built, the router wired, and the exact JSON body Render's health check
and the demo page's warm-up ping both depend on.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz_returns_ok() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_answers_head() -> None:
    """What an uptime monitor sends by default. A 405 here would have the
    monitor report a healthy service as down."""
    assert client.head("/healthz").status_code == 200
