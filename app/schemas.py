"""The three data shapes that cross module boundaries.

`AlertPayload` is what TradingView sends us. `MarketContext` is what Binance
gives us. `SignalBrief` is what Claude must produce. Nothing else travels
between modules as a raw dict.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]


class AlertPayload(BaseModel):
    """An inbound TradingView alert.

    Extra keys are ignored: TradingView's payload is ours to read, not ours
    to control, and rejecting an unrecognised field would drop a real alert.

    Raises `pydantic.ValidationError` on a missing or empty required field,
    or a `bar_time` that is not a parseable timestamp.
    """

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(min_length=1, max_length=32)
    timeframe: str = Field(min_length=1, max_length=16)
    condition: str = Field(min_length=1, max_length=200)
    bar_time: datetime
    price: float | None = None


class MarketContext(BaseModel):
    """The two numbers `enrich.py` computes from Binance klines.

    Two, not a hundred OHLCV rows: the model reasons better about computed
    features than it does about arithmetic, and it costs fewer tokens.
    """

    volume_vs_20d_avg: float  # 1.0 = at the 20-day average, 2.5 = 2.5x it.
    pct_from_20ma: float  # +3.2 = 3.2% above the 20MA, -1.5 = below.


class SignalBrief(BaseModel):
    """Claude's structured output. Invariant #2 lives here.

    There is deliberately no field that could hold a recommendation. A model
    that wants to say "buy" has nowhere to put it, and `extra="forbid"` turns
    an invented `recommendation` key into a ValidationError rather than a
    silently dropped one — so the advice reaches a failing test, not Telegram.
    """

    model_config = ConfigDict(extra="forbid")

    severity: Literal["low", "medium", "high"]
    headline: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)]
    observations: list[ShortText] = Field(min_length=2, max_length=4)
