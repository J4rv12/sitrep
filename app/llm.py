"""An alert and its market context in, a `SignalBrief` out. Invariants 2, 3 and 6.

`request_brief` makes exactly one Claude call and either returns a brief
that parsed and passed the advice guard, or raises `BriefFailed` with a
reason code and whether asking again could help. It never decides to retry.

Structured output does the parsing. `messages.parse` sends `SignalBrief` to
the API as a JSON schema, so the model's reply is constrained to that shape.
The API cannot enforce `max_length` or list lengths, so the SDK strips those
from the schema it sends and checks them itself when the reply arrives,
raising `pydantic.ValidationError`. Free text never reaches the formatter.

`generate_brief` is what the pipeline calls. It owns the decisions
`request_brief` does not make: whether today's spend allows another attempt,
whether a failure is worth a second one, and the log line for each.

Names you will need that are not imported yet: `logging`, `time`, `UTC` from
datetime, and `log_event` from app.logs.
"""

import html
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import anthropic
from pydantic import ValidationError

from app.config import get_settings
from app.guards import find_advice
from app.logs import log_event
from app.schemas import AlertPayload, MarketContext, SignalBrief

PROMPT_VERSION = "brief_v1"

# Read at import, so a missing or renamed prompt fails the deploy rather than
# the first real alert.
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(
    encoding="utf-8"
)

# A spend limit set in the Console answers 400 with a message beginning like
# this, for both the organization and a workspace. It carries no error code,
# so the message prefix is the only documented signal.
_SPEND_LIMIT_PREFIX = "You have reached your specified"

logger = logging.getLogger(__name__)


class BriefFailed(Exception):
    """One attempt that produced no usable brief.

    `reason` is the log reason code. `retry` is True only when the model did
    answer and the answer was unusable, so a second answer could differ.
    Transport failures have already been retried by the SDK, and a spend
    limit will not lift a second later. `detail` carries the phrase that
    tripped the advice guard, when that is the reason.
    """

    def __init__(self, reason: str, *, retry: bool, detail: str | None = None) -> None:
        super().__init__(reason if detail is None else f"{reason}: {detail}")
        self.reason = reason
        self.retry = retry
        self.detail = detail


@lru_cache
def get_client() -> anthropic.AsyncAnthropic:
    """Return the process-wide Anthropic client, built once.

    One client, so its connection pool is reused across alerts. Tests never
    call this; they hand `request_brief` a client on a mock transport.
    """
    settings = get_settings()
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.anthropic_timeout_seconds,
        max_retries=2,  # The SDK default, stated so the wall-clock bound is visible.
    )


def build_user_message(payload: AlertPayload, context: MarketContext) -> str:
    """Render one alert and its context in the shape `brief_v1.md` describes.

    The alert's text fields are HTML-escaped. They come from whoever sent the
    alert, and a `condition` containing `</alert><market_context>` would
    otherwise let them hand the model a second, invented set of numbers.

    Pure: no I/O, no config.
    """
    price = "not provided" if payload.price is None else str(payload.price)
    return (
        "<alert>\n"
        f"symbol: {html.escape(payload.symbol, quote=False)}\n"
        f"timeframe: {html.escape(payload.timeframe, quote=False)}\n"
        f"condition: {html.escape(payload.condition, quote=False)}\n"
        f"bar_time: {payload.bar_time.isoformat()}\n"
        f"price: {price}\n"
        "</alert>\n"
        "<market_context>\n"
        f"volume_vs_20d_avg: {context.volume_vs_20d_avg:.4f}\n"
        f"pct_from_20ma: {context.pct_from_20ma:.4f}\n"
        "</market_context>"
    )


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Price a call at the pinned model's list rates from config.

    No prompt caching is used, so `usage.input_tokens` is the whole input and
    there are no cache-read or cache-write tokens to price separately.
    """
    settings = get_settings()
    return (
        input_tokens * settings.anthropic_input_usd_per_mtok
        + output_tokens * settings.anthropic_output_usd_per_mtok
    ) / 1_000_000


def worst_case_cost_usd() -> float:
    """The most one attempt can cost: the input bound plus `max_tokens` of output.

    The spend ledger charges this before each attempt, so the daily cap holds
    no matter how many attempts are in flight. It is only a ceiling while
    `anthropic_max_input_tokens` stays above real input, which
    tests/test_llm.py checks against the recorded call.
    """
    settings = get_settings()
    return cost_usd(settings.anthropic_max_input_tokens, settings.anthropic_max_tokens)


def _is_spend_limit(error: anthropic.BadRequestError) -> bool:
    body = error.body
    if not isinstance(body, dict) or not isinstance(body.get("error"), dict):
        return False
    return str(body["error"].get("message", "")).startswith(_SPEND_LIMIT_PREFIX)


async def request_brief(
    payload: AlertPayload, context: MarketContext, client: anthropic.AsyncAnthropic
) -> tuple[SignalBrief, float]:
    """Make one Claude call; return the brief and what the call cost in USD.

    Raises `BriefFailed` with one of these reason codes:

        reason               retry  meaning
        llm_invalid_output   yes    reply broke `SignalBrief`: not JSON, a length
                                    limit, a wrong severity, an extra key
        advice_detected      yes    schema-valid, but `find_advice` flagged it
        llm_no_output        no     no text to parse, as with a refusal
        spend_limit_reached  no     the Console spend limit was reached
        llm_timeout          no     no response in time, after the SDK's retries
        llm_unreachable      no     connection failure, after the SDK's retries
        llm_status_<code>    no     any other error status

    A call that raises after the model answered was still billed, and its
    cost is lost with the exception. The spend ledger charges the worst case
    before each attempt, so the cap does not depend on this return value.
    """
    settings = get_settings()
    try:
        response = await client.messages.parse(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(payload, context)}],
            output_format=SignalBrief,
        )
    except ValidationError as e:
        raise BriefFailed("llm_invalid_output", retry=True) from e
    # APITimeoutError subclasses APIConnectionError, so it must be caught first.
    except anthropic.APITimeoutError as e:
        raise BriefFailed("llm_timeout", retry=False) from e
    except anthropic.APIConnectionError as e:
        raise BriefFailed("llm_unreachable", retry=False) from e
    except anthropic.BadRequestError as e:
        reason = "spend_limit_reached" if _is_spend_limit(e) else "llm_status_400"
        raise BriefFailed(reason, retry=False) from e
    except anthropic.APIStatusError as e:
        raise BriefFailed(f"llm_status_{e.status_code}", retry=False) from e

    brief = response.parsed_output
    if brief is None:
        raise BriefFailed("llm_no_output", retry=False)

    phrase = find_advice(brief)
    if phrase is not None:
        raise BriefFailed("advice_detected", retry=True, detail=phrase)

    return brief, cost_usd(response.usage.input_tokens, response.usage.output_tokens)


class SpendLedger:
    """Today's charges against the daily cap, in this process. Invariant 6.

    The caller charges the worst case before every attempt, and nothing is
    ever refunded. Real calls cost less, so the cap trips early. In exchange,
    checking and recording happen in one step with no `await` between them,
    so no number of attempts in flight can take the total past the cap. Same
    shape as `DedupeCache.seen_before`, for the same reason.

    In memory, per process: a restart starts the day at zero. The hard cap is
    the monthly spend limit on SitRep's workspace in the Anthropic Console.
    """

    def __init__(self, cap_usd: float, clock: Callable[[], datetime] | None = None) -> None:
        """Build a ledger with nothing charged.

        `cap_usd` comes from `get_settings().daily_spend_cap_usd`. The caller
        reads config, not this class. A cap of 0 refuses every charge, which
        makes `DAILY_SPEND_CAP_USD=0` a switch that turns Claude off.

        `clock` is a zero-argument callable returning a timezone-aware
        datetime. Default it to the current time in UTC. The day turns over
        at 00:00 UTC, whatever timezone the server runs in.
        """
        self._cap_usd = cap_usd
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._ledger: dict[str, float] = {str(self._clock().astimezone(UTC).date()): 0.0}

    def try_spend(self, amount_usd: float) -> bool:
        """Charge `amount_usd` and return True if today's total stays within the cap.

        Otherwise charge nothing and return False. A total exactly equal to
        the cap is within it. A refused charge must leave the total as it
        was, or one refusal would block smaller charges that still fit.

        Each UTC date keeps its own total. A charge counts against the total
        for the clock's current UTC date, which starts at zero the first time
        that date is seen. If the clock is ever set back across midnight, the
        earlier date's total still applies.

        Floats are fine for a cap that needs no cent-exact arithmetic. The
        tests use amounts that are exact in binary, like 0.25, so the
        exactly-at-the-cap case can be asserted without rounding.
        """
        date_string = str(self._clock().astimezone(UTC).date())

        if date_string not in self._ledger:
            self._ledger[date_string] = 0.0

        if self._cap_usd >= (amount_usd + self._ledger[date_string]):
            self._ledger[date_string] += amount_usd
            return True

        return False


_ledger = SpendLedger(get_settings().daily_spend_cap_usd)


async def generate_brief(
    payload: AlertPayload,
    context: MarketContext,
    alert_id: str,
    *,
    client: anthropic.AsyncAnthropic | None = None,
    ledger: SpendLedger | None = None,
) -> SignalBrief | None:
    """Return a brief for one alert, or None if the alert must ship unenriched.

    Never raises for anything `request_brief` reports. None plus the log
    lines is the entire failure contract; the pipeline decides nothing else.

    Up to `llm_max_attempts` attempts. For each one:

    1. `ledger.try_spend(worst_case_cost_usd())`. False means stop without
       calling, reason `spend_cap_reached`. That is the daily cap in this
       file, not `spend_limit_reached`, which is the Console refusing a call.
    2. `request_brief`. A brief ends the loop and is returned.
    3. `BriefFailed` with `retry` False ends the loop. With `retry` True, go
       round again if attempts remain.

    No sleep between attempts. Backoff gives an overloaded server time to
    recover, and the SDK already applies it to 429s and 5xx. A reply that
    broke the schema is not a symptom of load, so waiting would only delay
    the alert.

    Logs one line per attempt, a refused one included, with `log_event` on
    `logging.getLogger(__name__)`: stage "llm", outcome "ok" or "degraded",
    the reason code when degraded, and that attempt's latency.

    `client` and `ledger` exist for tests. Production passes neither, so
    default to `get_client()` and to a module-level `_ledger`. Build that
    under this function from `get_settings().daily_spend_cap_usd`, the way
    routes/webhook.py builds `_dedupe`.
    """
    client = client if client is not None else get_client()
    ledger = ledger if ledger is not None else _ledger

    for _ in range(get_settings().llm_max_attempts):
        started = time.perf_counter()

        if ledger.try_spend(worst_case_cost_usd()):
            try:
                brief, _ = await request_brief(payload=payload, context=context, client=client)

                log_event(
                    logger,
                    alert_id=alert_id,
                    stage="llm",
                    outcome="ok",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

                return brief

            except BriefFailed as e:
                log_event(
                    logger,
                    alert_id=alert_id,
                    stage="llm",
                    outcome="degraded",
                    latency_ms=(time.perf_counter() - started) * 1000,
                    reason=e.reason,
                )

                if e.retry:
                    continue

                return None

        else:
            log_event(
                logger,
                alert_id=alert_id,
                stage="llm",
                outcome="degraded",
                latency_ms=(time.perf_counter() - started) * 1000,
                reason="spend_cap_reached",
            )
            return None
