"""The advice guard: what it must catch, and what it must let through.

Both lists matter equally. A guard that flags everything passes the first
half and turns every alert into a fallback; one that flags nothing passes
the second. Each report below shares a stem with an advice case above it,
which is where a substring match or a careless word boundary gives itself
away.

These sentences are written for this test, not recorded model output. The
only numbers in them, 0.8x and -0.9%, are what `compute_context` produces
from the recorded BTCUSDT fixture, rounded the way the prompt asks.
"""

import pytest

from app.guards import find_advice
from app.schemas import SignalBrief

CLEAN_HEADLINE = "BTCUSDT close above 20MA on below-average volume"
CLEAN_OBSERVATIONS = [
    "Volume is 0.8x its 20-day average.",
    "Price is 0.9% below its 20-day average.",
]


def brief_with(text: str, field: str) -> SignalBrief:
    """A clean brief with `text` as its headline, or appended as its last observation.

    Last, not first: a guard that checks only `observations[0]` must fail.
    """
    if field == "headline":
        return SignalBrief(severity="low", headline=text, observations=CLEAN_OBSERVATIONS)
    return SignalBrief(
        severity="low", headline=CLEAN_HEADLINE, observations=[*CLEAN_OBSERVATIONS, text]
    )


FIELDS = ["headline", "observation"]

ADVICE = [
    pytest.param("Strong buy setup on BTCUSDT.", id="buy"),
    pytest.param("Sell into this strength while volume is thin.", id="sell"),
    pytest.param("Hold the position while price stays above the average.", id="hold"),
    pytest.param("Short the bounce into the average.", id="short"),
    pytest.param("Consider entering on a pullback to the average.", id="consider"),
    pytest.param("A clean entry if volume picks up.", id="entry"),
    pytest.param("Traders should wait for a daily close above the average.", id="should"),
    pytest.param("Wait for volume to confirm the move.", id="wait-for"),
    pytest.param("Look to add on any dip toward the average.", id="look-to"),
    pytest.param("Take profit into the move.", id="take-profit"),
    pytest.param("Keep a stop loss just under the average.", id="stop-loss"),
    pytest.param("The next target is the prior swing high.", id="target"),
    pytest.param("Exit if price loses the average.", id="exit"),
    pytest.param("We recommend staying flat until volume returns.", id="recommend"),
    pytest.param("STRONG BUY SIGNAL", id="uppercase"),
    # Added in review. Each got past the first implementation.
    pytest.param("Keep a stop-loss just under the average.", id="stop-loss-hyphen"),
    pytest.param("Take-profit into the move.", id="take-profit-hyphen"),
    pytest.param("Recommended: stay flat until volume returns.", id="recommended"),
    pytest.param("Targets sit at the prior swing high.", id="targets"),
    pytest.param("Exiting here locks in the move.", id="exiting"),
    pytest.param("Go long above the average.", id="go-long"),
]

REPORTS = [
    pytest.param("Buying interest is thin: volume is 0.8x its 20-day average.", id="buying"),
    pytest.param("Sellers have not stepped up; volume is 0.8x its 20-day average.", id="sellers"),
    pytest.param("Price is holding 0.9% below its 20-day average.", id="holding"),
    pytest.param("Volume has stopped expanding.", id="stopped"),
    pytest.param("The short-term move has little participation behind it.", id="short-term"),
    pytest.param("Price sits 0.9% below its longer-term average.", id="longer-term"),
    pytest.param("Close above 20MA fired on the 1h chart; context uses daily bars.", id="restated"),
]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("text", ADVICE)
def test_advice_is_caught(text: str, field: str) -> None:
    found = find_advice(brief_with(text, field))

    assert found is not None, f"advice got through in the {field}: {text!r}"
    assert found.lower() in text.lower(), f"returned {found!r}, which is not in {text!r}"


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("text", REPORTS)
def test_market_description_is_not_advice(text: str, field: str) -> None:
    found = find_advice(brief_with(text, field))

    assert found is None, f"flagged {found!r} in a plain report: {text!r}"


def test_clean_brief_passes() -> None:
    brief = SignalBrief(severity="low", headline=CLEAN_HEADLINE, observations=CLEAN_OBSERVATIONS)

    assert find_advice(brief) is None
