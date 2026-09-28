"""Token verification contract.

conftest.py sets WEBHOOK_TOKEN to "test-webhook-token-at-least-32-chars" before
anything imports app.config, so that is the secret under test throughout this
file.

The assertions use `is True` / `is False` rather than `assert token_is_valid(...)`.
A truthy return is not good enough here: this value decides whether a request
is authenticated, and "returns something truthy" is a weaker contract than
"returns a bool" when the next person refactors it.
"""

import pytest

from app.security import RateLimiter, token_is_valid
from conftest import FakeClock

VALID = "test-webhook-token-at-least-32-chars"


def test_correct_token_is_accepted() -> None:
    assert token_is_valid(VALID) is True


@pytest.mark.parametrize(
    ("supplied", "case"),
    [
        (VALID[:-1] + "S", "final character differs"),
        (VALID[:-1], "a correct prefix, one character short"),
        (VALID + "-extra", "correct token plus a suffix"),
        ("", "empty string"),
        (None, "no token key in the body at all"),
        (12345, "a JSON number rather than a string"),
        ([VALID], "the correct token, wrapped in a list"),
        (VALID.replace("e", "ë", 1), "non-ASCII"),
    ],
)
def test_bad_tokens_are_rejected(supplied: object, case: str) -> None:
    assert token_is_valid(supplied) is False, case


# --- RateLimiter: yours -----------------------------------------------------
#
# Small numbers throughout: a limit of 2 or 3 and a 60-second window show the
# same behaviour as 20 an hour, and a failure is readable at a glance.


def test_allows_up_to_the_limit_then_refuses() -> None:
    limiter = RateLimiter(limit=3, window_seconds=60, clock=FakeClock())

    assert [limiter.allow("1.2.3.4") for _ in range(4)] == [True, True, True, False]


def test_limit_of_zero_refuses_everyone_and_keeps_nothing() -> None:
    """DAILY_SPEND_CAP_USD=0 turns Claude off. DEMO_RATE_LIMIT_PER_HOUR=0 must
    refuse every request the same way, a key's first included, and a refusal
    must not leave a key behind with nothing to ever remove it."""
    limiter = RateLimiter(limit=0, window_seconds=60, clock=FakeClock())

    assert limiter.allow("1.2.3.4") is False
    assert len(limiter) == 0


def test_window_rolls_rather_than_resetting() -> None:
    """Requests at 0 s and 30 s. At 60 s the first has aged out and exactly one
    more fits; the one from 30 s still counts. A fixed window would either
    refuse at 60 s or let two through."""
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
    assert limiter.allow("1.2.3.4") is True
    clock.advance(30)
    assert limiter.allow("1.2.3.4") is True
    clock.advance(29)
    assert limiter.allow("1.2.3.4") is False, "59 s: both requests still count"

    clock.advance(1)

    assert limiter.allow("1.2.3.4") is True, "60 s: the first request stopped counting"
    assert limiter.allow("1.2.3.4") is False, "the one from 30 s still counts"


def test_refused_request_is_not_recorded() -> None:
    """A refusal at 30 s must not push the wait out to 90 s."""
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    assert limiter.allow("1.2.3.4") is True
    clock.advance(30)
    assert limiter.allow("1.2.3.4") is False

    clock.advance(30)

    assert limiter.allow("1.2.3.4") is True


def test_expired_requests_stop_counting_for_every_key() -> None:
    """A at 0 s, B at 1 s and 2 s, A at 50 s. At 63 s both of B's requests
    are over a minute old, so B has used none of its 2, even though B sits
    behind A, which is still active."""
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
    limiter.allow("A")
    clock.advance(1)
    limiter.allow("B")
    clock.advance(1)
    limiter.allow("B")
    clock.advance(48)
    limiter.allow("A")
    clock.advance(13)

    assert limiter.allow("B") is True, "B was refused for requests that no longer count"


def test_keys_are_counted_separately() -> None:
    limiter = RateLimiter(limit=1, window_seconds=60, clock=FakeClock())
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False

    assert limiter.allow("5.6.7.8") is True


def test_idle_keys_are_evicted_and_active_ones_kept() -> None:
    """A at 0 s, B at 10 s, A again at 50 s. At 75 s B has been idle for more
    than a window and must be gone. A must not, though A arrived first: its
    request at 50 s still counts."""
    clock = FakeClock()
    limiter = RateLimiter(limit=5, window_seconds=60, clock=clock)
    limiter.allow("A")
    clock.advance(10)
    limiter.allow("B")
    clock.advance(40)
    limiter.allow("A")
    clock.advance(25)

    limiter.allow("C")

    assert len(limiter) == 2, "expected A and C; B was idle for more than a window"
