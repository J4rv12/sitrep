"""Liveness endpoint.

Three callers: Render's health check, the demo page's warm-up ping on load,
and the uptime monitor whose ping every 5 minutes keeps a free-tier service
from sleeping (CLAUDE.md section 5).

All need this to be cheap. It touches no network and no config, so it stays
honest about one thing only: this process is up and serving.
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


# HEAD as well as GET. Uptime monitors check with HEAD by default, and FastAPI
# does not add it to a GET route: a GET-only /healthz answers 405, and the
# monitor reports a healthy service as down.
@router.api_route("/healthz", methods=["GET", "HEAD"])
async def healthz() -> dict[str, str]:
    """Return `{"status": "ok"}`. Never fails; if the process is down there
    is simply no response."""
    return {"status": "ok"}
