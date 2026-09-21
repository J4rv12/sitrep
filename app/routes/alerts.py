"""The alert feed: the last 50 alerts and what became of each.

`GET /alerts`, newest first, behind the webhook's token sent in the
`X-SitRep-Token` header. A GET has no body to carry it, and a query string
would write it into Render's access log.

Private because one deployment is one process with one feed. Every alert the
webhook accepts lands here, and a `condition` string is a trading strategy in
words. The Phase 5 demo page does not read this list.

In memory, per CLAUDE.md section 4. A restart empties it, and so does Render's
free tier going to sleep after 15 idle minutes. That is fine for what the feed
is for — recent outcomes without opening Render's logs — and a database on
free-tier disk would lose the same entries while looking like it kept them.

`maxlen` is the memory bound. A full deque with a maxlen drops its oldest
entry on every append, so the feed cannot grow however long the process lives.
"""

from collections import deque

from fastapi import APIRouter, HTTPException, Request

from app.schemas import FeedEntry
from app.security import token_is_valid

FEED_SIZE = 50
TOKEN_HEADER = "X-SitRep-Token"

router = APIRouter(tags=["alerts"])

_feed: deque[FeedEntry] = deque(maxlen=FEED_SIZE)


def record_alert(entry: FeedEntry) -> None:
    """Add one processed alert, dropping the oldest once there are `FEED_SIZE`.

    Never raises. It runs after `process_alert`'s net, like the Telegram send.
    """
    _feed.append(entry)


@router.get("/alerts")
async def list_alerts(request: Request) -> list[FeedEntry]:
    """Return the feed, newest first; 401 without the right `X-SitRep-Token`.

    The header is read by its literal name rather than through FastAPI's
    `Header()`, which derives the name from the parameter's spelling. This
    way, searching the repo for `X-SitRep-Token` finds the check.
    """
    if not token_is_valid(request.headers.get(TOKEN_HEADER)):
        raise HTTPException(status_code=401, detail="Token missing or wrong")
    return list(reversed(_feed))
