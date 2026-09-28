"""The two guards: who may send alerts, and how often anyone may run the demo.

`token_is_valid` authenticates the webhook and the feed. `RateLimiter` caps
demo requests per client IP, because the demo has no token to check.

The token travels in the JSON body, not a header. TradingView's alert dialog
lets you set the URL and the message text and nothing else, so a header check
would lock out the only client this service exists for. The body is static
text, which is all a static token needs. Not the URL: query strings are
written to Render's access log and Cloudflare's, and a secret in a log is a
secret you cannot rotate quietly.

This module knows nothing about FastAPI on purpose. It answers questions — is
this token correct, may this key go again — and the route decides what HTTP
status each answer deserves.
"""

import hmac
import time
from collections import deque
from collections.abc import Callable

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


class RateLimiter:
    """At most `limit` requests per key in any rolling `window_seconds`.

    Invariant 6's guard on the demo. The key is the client IP; which header
    that comes from is the route's decision, not this class's.

    Rolling, not fixed. A fixed window resets at the top of the hour, so 20
    requests at 10:59 and 20 more at 11:00 make 40 in two minutes. Here a
    request stops counting exactly `window_seconds` after it was recorded, so
    the 21st waits until the first is an hour old.

    Same shape as `DedupeCache`, for the same reasons: one instance per
    process, built by the route from config; checking and recording in one
    step; an injected clock; and nothing kept after it stops mattering.

    Names you will need that are not imported yet: `time`, and `deque` from
    collections, which pops from the left in constant time.
    """

    def __init__(
        self, limit: int, window_seconds: float, clock: Callable[[], float] | None = None
    ) -> None:
        """Build a limiter that has seen no requests.

        `limit` comes from `get_settings().demo_rate_limit_per_hour` and
        `window_seconds` is 3600; the route reads config, not this class.

        `clock` defaults to `time.monotonic`, for the reason in
        `DedupeCache.__init__`: a wall clock that jumps backwards would bring
        old requests back into the window.
        """
        self._limit = limit
        self._window_seconds = window_seconds
        self._clock = clock if clock is not None else time.monotonic
        self._count_per_key: dict[str, int] = {}
        self._time_and_key: deque[tuple[float, str]] = deque()

    def allow(self, key: str) -> bool:
        """Return True and record the request if `key` may go ahead; otherwise False.

        **This method mutates**, like `DedupeCache.seen_before`: a True
        answer is also a recorded request. Two calls a request apart could
        both see room for one more and both go through.

        A refused request records nothing. If it counted, a page retrying
        every few seconds while blocked would keep itself blocked forever.

        Nothing may outlive its usefulness. A key whose every request is
        older than the window is dropped, not kept at zero: anyone who can
        reach the demo adds a key, and a dict that only grows is a memory
        leak.

        The queue holds requests, not keys. Each request is appended once and
        the clock only moves forward, so the oldest is always at the front:
        expiring is one loop from the front that stops at the first live
        request, and a key leaves when its count reaches zero. Keys come
        back; requests never do, which is why this needs no reordering.
        """
        now = self._clock()

        while self._time_and_key and (self._time_and_key[0][0] + self._window_seconds <= now):
            _, key_value = self._time_and_key.popleft()
            self._count_per_key[key_value] -= 1

            if self._count_per_key[key_value] == 0:
                del self._count_per_key[key_value]

        if self._count_per_key.get(key, 0) < self._limit:
            self._time_and_key.append((now, key))
            self._count_per_key[key] = self._count_per_key.get(key, 0) + 1
            return True
        else:
            return False

    def __len__(self) -> int:
        """Return the number of keys currently held.

        Exists so a test can prove idle keys are evicted.
        """
        return len(self._count_per_key)
