"""Liveness endpoint.

Two callers: Render's health check, and the demo page's warm-up ping on load
(CLAUDE.md section 5 — it wakes a sleeping free-tier service while the
visitor reads the intro).

Both need this to be cheap. It touches no network and no config, so it stays
honest about one thing only: this process is up and serving.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return `{"status": "ok"}`. Never fails; if the process is down there
    is simply no response."""
    return {"status": "ok"}
