"""Authentication for the webhook. One function, one decision.

The token travels in the JSON body, not a header. TradingView's alert dialog
lets you set the URL and the message text and nothing else, so a header check
would lock out the only client this service exists for. The body is static
text, which is all a static token needs. Not the URL: query strings are
written to Render's access log and Cloudflare's, and a secret in a log is a
secret you cannot rotate quietly.

This module knows nothing about FastAPI on purpose. It answers one question —
is this token correct — and the route decides what HTTP status that deserves.
"""

import hmac

from app.config import get_settings


def token_is_valid(supplied: object) -> bool:
    """Return True if `supplied` matches the configured webhook token.

    The expected value is `get_settings().webhook_token` — read it in here,
    do not accept it as a parameter. A function that compares against
    whatever the caller passes can be handed the wrong secret by the next
    person who calls it.

    `supplied` is typed `object` deliberately. It comes from `json.loads` on
    a request body nobody has authenticated yet, so it may be a string, a
    number, a list, a dict, or missing entirely. Only a string equal to the
    configured token is valid; everything else is not.

    Never raises, and that is a requirement rather than a nicety. A caller who
    can make this function raise has turned a bad token into a 500, and a
    guess that 500s while others 401 is a guess that told them something.
    """
    if isinstance(supplied, str):
        return hmac.compare_digest(supplied.encode(), get_settings().webhook_token.encode())

    return False
