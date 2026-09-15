"""An alert and its market context in, a `SignalBrief` out. Invariants 2, 3 and 6.

`request_brief` makes exactly one Claude call and either returns a brief
that parsed and passed the advice guard, or raises `BriefFailed` with a
reason code and whether asking again could help. It never decides to retry.

Structured output does the parsing. `messages.parse` sends `SignalBrief` to
the API as a JSON schema, so the model's reply is constrained to that shape.
The API cannot enforce `max_length` or list lengths, so the SDK strips those
from the schema it sends and checks them itself when the reply arrives,
raising `pydantic.ValidationError`. Free text never reaches the formatter.
"""

import html
from functools import lru_cache
from pathlib import Path

import anthropic
from pydantic import ValidationError

from app.config import get_settings
from app.guards import find_advice
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
