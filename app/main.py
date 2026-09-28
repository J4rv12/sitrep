"""FastAPI application object and router wiring."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logs import configure_logging
from app.routes import alerts, demo, health, webhook

# Eager, at import time, on purpose. Uvicorn imports this module to find
# `app`; if a required setting is missing this raises here, the process exits
# non-zero, and Render fails the deploy while the previous version keeps
# serving. Left lazy, the same mistake boots green and 500s on the first
# real alert.
configure_logging(get_settings().log_level)

app = FastAPI(title="SitRep", version="0.1.0")

# Lets the demo page read `/healthz` and the demo stream. The page sends only
# "simple" requests (no custom headers, no JSON body), so the browser sends
# no preflight and each click costs one round trip. No custom headers are
# allowed, so a page that tried to send `X-SitRep-Token` to read the feed is
# refused at the preflight. Starlette always allows `Content-Type`, so the
# page could post JSON to `/webhook`; without the token that is a 401. The
# token protects the webhook, not CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().demo_allowed_origins,
    allow_methods=["GET", "POST"],
)

app.include_router(health.router)
app.include_router(webhook.router)
app.include_router(alerts.router)
app.include_router(demo.router)
