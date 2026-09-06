"""Alert identity and the dedupe window.

Two contracts. `compute_alert_id` must collapse retries of one alert onto one
id and keep genuinely different alerts apart. `DedupeCache` must answer True
inside the window, False outside it, and not grow without bound.

The cache tests drive a fake clock rather than sleeping. A real five-minute
expiry test would blow CLAUDE.md's 30-second suite budget on its own.
"""

import pytest

from app.dedupe import DedupeCache, compute_alert_id
from app.schemas import AlertPayload

BASE = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "condition": "close above 20MA",
    "bar_time": "2026-09-04T10:00:00Z",
}


def payload(**overrides: object) -> AlertPayload:
    return AlertPayload(**(BASE | overrides))


class FakeClock:
    """A monotonic clock you control. Starts at an arbitrary non-zero value
    so an implementation that treats 0.0 as "never seen" fails here."""

    def __init__(self, now: float = 10_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- compute_alert_id -------------------------------------------------------


def test_same_payload_gives_same_id() -> None:
    assert compute_alert_id(payload()) == compute_alert_id(payload())


def test_id_is_a_sha256_hex_digest() -> None:
    alert_id = compute_alert_id(payload())

    assert len(alert_id) == 64
    assert set(alert_id) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "ETHUSDT"),
        ("timeframe", "4h"),
        ("condition", "close below 20MA"),
        ("bar_time", "2026-09-04T11:00:00Z"),
    ],
)
def test_each_identifying_field_changes_the_id(field: str, value: str) -> None:
    assert compute_alert_id(payload(**{field: value})) != compute_alert_id(payload())


def test_price_does_not_change_the_id() -> None:
    """A retry can carry a different price and is still the same alert."""
    assert compute_alert_id(payload(price=64000.0)) == compute_alert_id(payload(price=64100.0))


def test_field_boundaries_are_respected() -> None:
    """Same characters, different fields. Without a separator these collide."""
    split_one = payload(symbol="BTC", timeframe="USDT1h")
    split_two = payload(symbol="BTCUSDT", timeframe="1h")

    assert compute_alert_id(split_one) != compute_alert_id(split_two)


def test_same_instant_in_a_different_offset_gives_one_id() -> None:
    """10:00Z and 15:30+05:30 are the same bar. One alert, one id."""
    utc = payload(bar_time="2026-09-04T10:00:00Z")
    kolkata = payload(bar_time="2026-09-04T15:30:00+05:30")

    assert compute_alert_id(utc) == compute_alert_id(kolkata)


# --- DedupeCache ------------------------------------------------------------


def test_first_sighting_is_not_a_duplicate() -> None:
    cache = DedupeCache(ttl_seconds=300, clock=FakeClock())

    assert cache.seen_before("a" * 64) is False


def test_second_sighting_within_ttl_is_a_duplicate() -> None:
    clock = FakeClock()
    cache = DedupeCache(ttl_seconds=300, clock=clock)
    alert_id = "a" * 64

    cache.seen_before(alert_id)
    clock.advance(299)

    assert cache.seen_before(alert_id) is True


def test_sighting_after_ttl_expiry_is_not_a_duplicate() -> None:
    clock = FakeClock()
    cache = DedupeCache(ttl_seconds=300, clock=clock)
    alert_id = "a" * 64

    cache.seen_before(alert_id)
    clock.advance(301)

    assert cache.seen_before(alert_id) is False


def test_expiry_is_measured_from_first_sighting() -> None:
    """A retry at t+200 must not extend the window to t+500.

    TradingView can retry for longer than the window. If each sighting reset
    the timer, a persistent retry loop would suppress the *next* genuine
    alert on the same condition.
    """
    clock = FakeClock()
    cache = DedupeCache(ttl_seconds=300, clock=clock)
    alert_id = "a" * 64

    cache.seen_before(alert_id)
    clock.advance(200)
    cache.seen_before(alert_id)
    clock.advance(101)

    assert cache.seen_before(alert_id) is False


def test_different_ids_do_not_collide() -> None:
    cache = DedupeCache(ttl_seconds=300, clock=FakeClock())

    assert cache.seen_before("a" * 64) is False
    assert cache.seen_before("b" * 64) is False
    assert cache.seen_before("a" * 64) is True


def test_expired_entries_are_evicted_not_just_ignored() -> None:
    """Answering correctly is not enough — the memory has to come back."""
    clock = FakeClock()
    cache = DedupeCache(ttl_seconds=300, clock=clock)

    for n in range(50):
        cache.seen_before(f"{n:064d}")
    assert len(cache) == 50

    clock.advance(301)
    cache.seen_before("f" * 64)

    assert len(cache) == 1
