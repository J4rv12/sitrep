"""The alert endpoint. Invariant 1 lives here.

`POST /webhook`. Five steps, in this order, and the order is the design:

    parse -> authenticate -> validate -> dedupe -> enqueue -> 202

The contract:

| Situation                                    | Status | Body                              |
|----------------------------------------------|--------|-----------------------------------|
| Accepted, queued for processing              | 202    | {"status": "accepted", "alert_id"}|
| Already seen inside the dedupe window        | 200    | {"status": "duplicate", "alert_id"}|
| Body is not valid JSON, or not a JSON object | 400    | FastAPI's {"detail": ...}         |
| Token missing or wrong                       | 401    | FastAPI's {"detail": ...}         |
| Token good, alert fields invalid             | 422    | FastAPI's {"detail": ...}         |

Why parse before authenticating: the token is a key in the JSON body, so
there is nothing to compare until the body is parsed. An unparseable body
therefore gets a 400 without ever being authenticated, which is the one
ordering concession the design makes.

Why authenticate before validating: an unauthenticated caller must not be
able to probe your schema. Wrong token plus a malformed alert is a 401, not
a 422 — they learn that the token was wrong and nothing else.

Why the handler is `async def`: it runs on the event loop, so it cannot be
interrupted between the dedupe check and the dedupe write. A plain `def`
would be handed to a threadpool, where two simultaneous retries of the same
alert can both pass the check and both deliver.

`get_market_context` and `generate_brief` are imported by name on purpose.
tests/test_webhook.py replaces `webhook.get_market_context` and
`webhook.generate_brief`, which is what keeps the suite away from Binance and
Anthropic. Reached through their modules instead, they would escape the patch.
"""

import logging
import time
from json import JSONDecodeError

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import get_settings
from app.dedupe import DedupeCache, compute_alert_id
from app.enrich import get_market_context
from app.llm import generate_brief
from app.logs import log_event
from app.schemas import AlertPayload, SignalBrief
from app.security import token_is_valid

router = APIRouter(tags=["webhook"])

# Process-local, built once at import. One instance per Render process; a
# restart empties it and costs at most one duplicate message.
_dedupe = DedupeCache(ttl_seconds=get_settings().dedupe_ttl_seconds)

logger = logging.getLogger(__name__)


async def process_alert(payload: AlertPayload, alert_id: str) -> SignalBrief | None:
    """Run the pipeline for one accepted alert; return its brief, or None.

    This is the slow half — Binance, then Claude, and from Phase 4 Telegram —
    several seconds end to end. It runs *after* the response has been sent,
    which is the entire reason invariant 1 holds. The background task
    ignores the return value; tests and Phase 4's delivery use it.

    In order:

    1. `get_market_context(payload.symbol, alert_id)`.
    2. None means Binance could not supply the numbers. Return None without
       calling Claude (invariant 4): a brief with nothing to report would
       have to be padded, and the model is not allowed to invent data.
    3. `generate_brief(payload, context, alert_id)` and return what it does.
       Its failures are its own to log; there is nothing to add here.

    Around all of it, the last-resort net. Both functions promise never to
    raise for an upstream failure, so anything that escapes them is a bug in
    our code — `compute_context` dividing by zero, say. After the 202 was
    sent, an exception here reaches nobody but stderr, and the alert is
    dropped. So catch `Exception`, log stage "pipeline", outcome "degraded",
    reason "internal_error" with `exc_info=True` so the traceback is in the
    line, and return None. `asyncio.CancelledError` is not an `Exception`
    and must pass through: it is how the server stops this task at shutdown,
    and swallowing it would keep the process alive with zombie tasks.

    This is the one broad `except` the manual allows (CLAUDE.md section 6).
    Nothing else in `app/` may copy it.
    """
    started = time.perf_counter()

    try:
        context = await get_market_context(payload.symbol, alert_id)

        if context is None:
            return None

        return await generate_brief(payload, context, alert_id)
    except Exception:
        log_event(
            logger,
            alert_id=alert_id,
            stage="pipeline",
            outcome="degraded",
            latency_ms=(time.perf_counter() - started) * 1000,
            reason="internal_error",
            exc_info=True,
        )

        return None


@router.post("/webhook", status_code=202)
async def receive_alert(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """Accept a TradingView alert, enqueue it, and return immediately.

    Steps, in order:

    1. Read the JSON body. `await request.json()` raises on malformed input;
       catch it and 400. A body that parses but is not a JSON object (a list,
       a bare string) is also a 400 — you cannot read a token out of it.
    2. Pull the "token" key out and hand it to `token_is_valid`. False is a
       401, and that is the end of the request.
    3. Build an `AlertPayload` from the body. `ValidationError` is a 422.
       `AlertPayload` ignores unknown keys, so the "token" key needs no
       special handling — it is simply not part of the model.
    4. `compute_alert_id`, then `_dedupe.seen_before`. True means this is a
       retry: return 200 with "duplicate" and enqueue nothing.
    5. `background_tasks.add_task(process_alert, payload, alert_id)`, then
       return 202 with "accepted".

    **Nothing in this function may open a socket, and nothing may `await`
    anything slow.** The only `await` here is `request.json()`, which reads a
    buffer that has already arrived. Every network call belongs in
    `process_alert`.

    Returns a `JSONResponse` in both success cases because they carry
    different status codes. A returned Response object overrides the
    decorator's `status_code=202`, which documents the primary path for
    OpenAPI and nothing more.
    """
    detail_400 = "Body is not valid JSON, or not a JSON object"

    try:
        body = await request.json()
    except JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=detail_400) from e

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail=detail_400)

    if not token_is_valid(body.get("token")):
        raise HTTPException(status_code=401, detail="Token missing or wrong")

    # To prevent the secret from leaking
    del body["token"]
    try:
        payload = AlertPayload.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail="Alert fields invalid") from e

    alert_id = compute_alert_id(payload)

    if _dedupe.seen_before(alert_id):
        return JSONResponse(content={"status": "duplicate", "alert_id": alert_id}, status_code=200)

    background_tasks.add_task(process_alert, payload, alert_id)
    return JSONResponse(content={"status": "accepted", "alert_id": alert_id}, status_code=202)
