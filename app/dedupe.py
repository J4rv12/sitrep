"""Invariant 5: the same alert must not be delivered twice.

Two pieces. `compute_alert_id` decides what "the same alert" means, and
`DedupeCache` remembers those ids for a few minutes.

No SQLite here, per CLAUDE.md section 4. Render's free tier has ephemeral
disk, so a database would silently lose its contents on every restart while
looking like it worked. A plain dict in memory loses its contents on restart
too, but it does so honestly, and a restart already means at most one
duplicate rather than a corrupt store.
"""

import hashlib
import time
from collections.abc import Callable
from datetime import UTC

from app.schemas import AlertPayload


def compute_alert_id(payload: AlertPayload) -> str:
    """Return the SHA-256 hex digest identifying this alert.

    The digest covers exactly four fields, joined with "|":
    `symbol`, `timeframe`, `condition`, `bar_time`.

    The separator is not decoration. Concatenated bare, symbol "BTC" with
    timeframe "USDT1h" and symbol "BTCUSDT" with timeframe "1h" produce the
    same bytes and therefore the same id, so one alert would silently
    suppress the other.

    `price` is deliberately excluded. It is what the market did, not what the
    alert is; two retries of one alert can carry different prices and must
    still collide.

    `bar_time` needs care. `2026-09-04T10:00:00Z` and
    `2026-09-04T15:30:00+05:30` are the same instant written two ways, and
    `datetime.isoformat()` returns a different string for each. The same bar
    must produce the same id no matter which offset the sender used, so
    normalise to UTC before formatting. `datetime.UTC` is the timezone to
    convert to.

    Pure: no clock, no config, no state. The same payload returns the same
    digest in any process, on any machine, forever.
    """
    symbol = payload.symbol
    timeframe = payload.timeframe
    condition = payload.condition
    bar_time = payload.bar_time.astimezone(UTC).isoformat()
    payload_processed = "|".join([symbol, timeframe, condition, bar_time])

    return hashlib.sha256(payload_processed.encode()).hexdigest()


class DedupeCache:
    """Alert ids seen within a rolling time window.

    One instance per process, created by the route. Not shared between
    Render instances, and emptied by a restart — both acceptable, because
    the window is five minutes and the cost of a miss is one duplicate
    message rather than a wrong one.
    """

    def __init__(self, ttl_seconds: int, clock: Callable[[], float] | None = None) -> None:
        """Build an empty cache.

        `ttl_seconds` comes from `get_settings().dedupe_ttl_seconds`. The
        caller reads config, not this class — it stays a plain data
        structure that a test can build with a two-second window.

        `clock` is a zero-argument callable returning seconds as a float.
        Default it to `time.monotonic`, not `time.time`. Wall-clock time
        jumps when NTP corrects the machine's drift; a backwards jump would
        make expired entries look fresh again. `time.monotonic` only ever
        moves forward, which is the only property this cache needs.

        Injecting it is what makes the expiry test finish instantly instead
        of sitting there for five real minutes.
        """
        self._ttl_seconds = ttl_seconds
        self._clock = clock if clock is not None else time.monotonic
        self._cache: dict[str, float] = {}

    def seen_before(self, alert_id: str) -> bool:
        """Return True if `alert_id` was recorded within the last `ttl_seconds`.

        **This method mutates.** A False answer records `alert_id` on the way
        out, so the second call with the same id returns True. Past tense in
        the name is doing real work: it reports the state *before* this call.

        One method rather than a separate `contains` and `record` on purpose.
        Two calls can be interleaved by a second request between them, and
        both would then see an empty cache and both would deliver. Checking
        and recording in one step makes that impossible.

        Expired entries must be dropped, not merely ignored. A dict that only
        grows is a memory leak on a process that stays up for weeks, and this
        one is fed by anyone who can reach the endpoint.
        """
        current_time = self._clock()

        while self._cache:
            alert = next(iter(self._cache))
            if (self._cache[alert] + self._ttl_seconds) < current_time:
                del self._cache[alert]
            else:
                break

        if alert_id in self._cache:
            return True

        self._cache[alert_id] = current_time

        return False

    def __len__(self) -> int:
        """Return the number of ids currently held.

        Exists so a test can prove expired entries are actually evicted.
        Unbounded growth is invisible from the outside otherwise: the cache
        keeps answering correctly right up until the process is killed for
        using too much memory.
        """
        return len(self._cache)
