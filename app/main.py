"""FastAPI application object and router wiring."""

from fastapi import FastAPI

from app.config import get_settings
from app.logs import configure_logging
from app.routes import alerts, health, webhook

# Eager, at import time, on purpose. Uvicorn imports this module to find
# `app`; if a required setting is missing this raises here, the process exits
# non-zero, and Render fails the deploy while the previous version keeps
# serving. Left lazy, the same mistake boots green and 500s on the first
# real alert.
configure_logging(get_settings().log_level)

app = FastAPI(title="SitRep", version="0.1.0")
app.include_router(health.router)
app.include_router(webhook.router)
app.include_router(alerts.router)
