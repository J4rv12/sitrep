"""The eval: 20 labelled alerts through the real pipeline, scored.

Two modes, because the numbers must be real and the run must be repeatable:

    make eval-record   Online and billed, about $0.05. Sends every case in
                       dataset.jsonl through `generate_brief` and the real
                       Claude, and saves each raw reply with how long it took
                       to recordings.jsonl.
    make eval          Offline and free. The same code path, but Claude's
                       part is answered from recordings.jsonl. Prints the
                       score.

A person then marks each saved reply as advice or not, in recordings.jsonl.
The guard cannot mark its own work: advice it missed, it also counts as clean.

`CaseResult` is what the pipeline did with one dataset case. `summarise`
turns a list of them into a `Summary`. Both are plain data with no I/O, so
the arithmetic can be tested without a recording.

Each case's market numbers come from Binance's public daily bars, through
`app.enrich.compute_context`: 22 rows ending on the case's `context_as_of`
day, so the latest price is that day's close.
"""

import asyncio
import hashlib
import json
import logging
import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import anthropic
import httpx2
from pydantic import ValidationError

from app.config import get_settings
from app.dedupe import compute_alert_id
from app.llm import SpendLedger, cost_usd, generate_brief
from app.schemas import AlertPayload, MarketContext

Outcome = Literal["rejected", "unenriched", "enriched", "crashed"]
Severity = Literal["low", "medium", "high"]


class Rate(NamedTuple):
    """A rate kept as its two counts, so "0 of 0" cannot pass for 100%."""

    hits: int
    total: int


@dataclass(frozen=True)
class Attempt:
    """One reply from Claude, as recorded."""

    schema_valid: bool  # It parsed into a SignalBrief.
    guard_fired: bool  # find_advice flagged it. False if it never parsed.
    marked_advice: bool | None  # A person's reading of it. None: not read yet.
    latency_ms: float  # Measured when the real call was made.
    cost_usd: float  # Its recorded tokens at list price.


@dataclass(frozen=True)
class CaseResult:
    """What the pipeline did with one case, beside what its label expected."""

    case_id: str
    expected_outcome: Outcome
    expected_severity: Severity | None
    outcome: Outcome  # "crashed": the code raised where it should have answered.
    severity: Severity | None  # The delivered brief's. None when there is no brief.
    attempts: tuple[Attempt, ...] = ()  # Empty when Claude was never called.
    reasons: tuple[str, ...] = ()  # Why it went as it did: reason codes, in order.


@dataclass(frozen=True)
class Summary:
    schema_valid: Rate
    severity_agreement: Rate
    guard_catch: Rate
    unmarked: int
    fallback: Rate
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    median_cost_usd: float | None


def percentile(values: list[float], p: int) -> float:
    """Return the `p`th percentile of `values` by nearest rank.

    Sort, then take the value at position ceil(p% of n), counting from 1. The
    answer is always a value that occurred. Interpolating would report a
    latency no call ever had. With 16 values the 95th is the largest.

    Raises `ValueError` on an empty list: there is no value to return.
    """
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    return ordered[math.ceil(p * len(ordered) / 100) - 1]


def summarise(results: list[CaseResult]) -> Summary:
    """Turn one eval run into the numbers `results.md` reports.

    Every rate comes back as counts, `Rate(hits, total)`, and is never
    divided here. What each one counts:

        schema_valid        Replies that parsed, out of all replies. Counts
                            replies, not alerts: an alert that was retried
                            has two.
        severity_agreement  Delivered briefs whose severity equals the label,
                            out of delivered briefs whose label has one. An
                            alert that shipped unenriched has no brief to
                            grade, so it is left out. It shows in `fallback`.
        guard_catch         Replies the guard fired on, out of replies that
                            parsed and were marked as advice. A reply that
                            did not parse never reaches the guard, so it is
                            not the guard's to catch. A clean reply the guard
                            fired on is a false alarm: not in this rate at
                            all.
        unmarked            Replies nobody has marked yet (`marked_advice` is
                            None), parsed or not. While this is above zero
                            `guard_catch` is unfinished.
        fallback            Alerts that shipped unenriched, out of alerts
                            delivered. Delivered means enriched or
                            unenriched; a rejected or crashed alert was not.

    Latency and cost are per alert, over the alerts that called Claude:

        latency_p50_ms      Each alert's attempts added up, then
        latency_p95_ms      `percentile(..., 50)` and `percentile(..., 95)`.
        median_cost_usd     The same for cost: attempts added up, then
                            `percentile(..., 50)`. A retry is billed, so it
                            counts.

    All three are None when no alert called Claude.

    Pure: no I/O, no clock. Never raises, an empty list included.
    """
    replies = [attempt for result in results for attempt in result.attempts]
    grades = [
        result
        for result in results
        if result.outcome == "enriched" and result.expected_severity is not None
    ]
    advice = [reply for reply in replies if reply.schema_valid and reply.marked_advice is True]
    delivered = [result for result in results if result.outcome in ("enriched", "unenriched")]
    latencies = [
        sum(reply.latency_ms for reply in result.attempts) for result in results if result.attempts
    ]
    costs = [
        sum(reply.cost_usd for reply in result.attempts) for result in results if result.attempts
    ]

    schema_valid = Rate(sum(reply.schema_valid for reply in replies), len(replies))
    severity_agreement = Rate(
        sum(result.severity == result.expected_severity for result in grades), len(grades)
    )
    guard_catch = Rate(sum(reply.guard_fired for reply in advice), len(advice))
    unmarked = sum(reply.marked_advice is None for reply in replies)
    fallback = Rate(sum(result.outcome == "unenriched" for result in delivered), len(delivered))
    latency_p50_ms = percentile(latencies, 50) if latencies else None
    latency_p95_ms = percentile(latencies, 95) if latencies else None
    median_cost_usd = percentile(costs, 50) if costs else None

    return Summary(
        schema_valid=schema_valid,
        severity_agreement=severity_agreement,
        guard_catch=guard_catch,
        unmarked=unmarked,
        fallback=fallback,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        median_cost_usd=median_cost_usd,
    )


# --- recording: a copy of every exchange with Claude ------------------------


def request_key(request: httpx2.Request) -> str:
    """Return a fingerprint of one request to Claude: model, prompt, alert, schema.

    A recorded reply answers one exact request. Replay sends the request
    again and compares fingerprints, so a reply recorded under an older
    prompt cannot be scored as the current prompt's.

    Keys are sorted before hashing: the same request must give the same
    fingerprint whatever order the SDK happens to write its fields in.
    """
    body = json.dumps(json.loads(request.content), sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


class Recorder(httpx2.AsyncBaseTransport):
    """Sits between the SDK and the network and keeps a copy of each exchange.

    `generate_brief` runs unchanged on a client built over this, so what is
    kept is what production would have received, a retry included. Latency
    runs from sending the request to holding the whole reply.
    """

    def __init__(self, inner: httpx2.AsyncBaseTransport) -> None:
        self._inner = inner
        self.exchanges: list[dict[str, object]] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        started = time.perf_counter()
        response = await self._inner.handle_async_request(request)
        body = await response.aread()
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            parsed: object = json.loads(body)
        except ValueError:  # An error page from a proxy, say. Kept as text.
            parsed = body.decode(errors="replace")
        self.exchanges.append(
            {
                "request_sha256": request_key(request),
                "status": response.status_code,
                "latency_ms": round(latency_ms, 1),
                "response": parsed,
            }
        )
        # `aread` has already unzipped the body. The copy handed on must not
        # still claim to be gzipped, or the SDK would try to unzip it again.
        headers = [
            (name, value)
            for name, value in response.headers.multi_items()
            if name.lower() not in ("content-encoding", "content-length")
        ]
        return httpx2.Response(response.status_code, headers=headers, content=body, request=request)


# --- replay: the saved replies, handed back in order ------------------------


class Replay:
    """Stands in for the network: answers each request with the next saved reply.

    Built from the lines of recordings.jsonl. Call `begin(case_id)` before
    each case; `served` then fills with the lines that case used, in order.

    A request it cannot answer honestly goes into `problems`, and the caller
    must stop the run when that is not empty. It cannot raise instead: the
    SDK turns an exception in the transport into a connection error, which
    `generate_brief` logs as `llm_unreachable` and returns None for. A stale
    recording would then score as an ordinary fallback.
    """

    def __init__(self, recorded: list[dict[str, object]]) -> None:
        self._by_case: dict[object, list[dict[str, object]]] = {}
        for line in recorded:
            self._by_case.setdefault(line["case"], []).append(line)
        self._case = ""
        self.served: list[dict[str, object]] = []
        self.problems: list[str] = []

    def begin(self, case_id: str) -> None:
        self._case = case_id
        self.served.clear()

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        saved = self._by_case.get(self._case, [])
        if len(self.served) == len(saved):
            return self._refuse("asked for a reply that was never recorded")
        line = saved[len(self.served)]
        if line["request_sha256"] != request_key(request):
            return self._refuse("the request has changed since it was recorded")
        self.served.append(line)
        return httpx2.Response(200, json=line["response"])

    def _refuse(self, why: str) -> httpx2.Response:
        self.problems.append(f"{self._case}: {why}")
        # A 400 because the SDK does not retry one. It would retry a 5xx,
        # sleeping between tries.
        error = {"type": "invalid_request_error", "message": why}
        return httpx2.Response(400, json={"type": "error", "error": error})


# --- one case through the pipeline ------------------------------------------

# What the reason code `generate_brief` logs for an attempt says about the
# reply it got: (it parsed, the guard fired). None is the code of a success.
# Any other code means no reply arrived at all.
_VERDICTS: dict[str | None, tuple[bool, bool]] = {
    None: (True, False),
    "advice_detected": (True, True),
    "llm_invalid_output": (False, False),
    "llm_no_output": (False, False),
}


def reply_cost(reply: dict[str, Any]) -> float:
    """Price one recorded reply from the token counts the API returned with it."""
    usage = reply["response"]["usage"]
    return cost_usd(usage["input_tokens"], usage["output_tokens"])


@contextmanager
def attempt_reasons() -> Iterator[list[str | None]]:
    """Collect the reason code `generate_brief` logs for each attempt.

    The eval reads the pipeline's verdict on a reply from the same log line
    production writes. It does not parse the reply or run the guard a second
    time, so it cannot disagree with what the pipeline did.
    """
    reasons: list[str | None] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            reasons.append(getattr(record, "reason", None))

    logger = logging.getLogger("app.llm")
    handler, level, propagate = Collect(), logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # A success is logged at INFO.
    logger.propagate = False  # Twenty cases would otherwise fill the terminal.
    try:
        yield reasons
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate


async def run_case(
    case: dict[str, Any],
    client: anthropic.AsyncAnthropic,
    ledger: SpendLedger,
    exchanges: list[dict[str, Any]],
) -> CaseResult:
    """Put one dataset case through the pipeline's own steps, in the webhook's order.

    Validate the body, compute the alert id, and call `generate_brief` if the
    case has market numbers. `exchanges` is the list the client's transport
    fills, a `Recorder`'s or a `Replay`'s; the caller empties it first.

    The broad `except` stands in for the web server, which answers 500 to
    anything the route lets escape. Here that is recorded as "crashed".
    """
    outcome: Outcome = "unenriched"
    severity, reasons = None, ()
    with attempt_reasons() as logged:
        try:
            payload = AlertPayload.model_validate(case["body"])
            alert_id = compute_alert_id(payload)
        except ValidationError:
            outcome = "rejected"
        except Exception as error:
            outcome, reasons = "crashed", (type(error).__name__,)
        else:
            if case["context"] is None:
                reasons = ("no_market_context",)
            else:
                context = MarketContext(**case["context"])
                brief = await generate_brief(
                    payload, context, alert_id, client=client, ledger=ledger
                )
                if brief is not None:
                    outcome, severity = "enriched", brief.severity
    replies = [exchange for exchange in exchanges if exchange["status"] == 200]
    verdicts = [_VERDICTS[reason] for reason in logged if reason in _VERDICTS]
    attempts = tuple(
        Attempt(
            schema_valid=parsed,
            guard_fired=fired,
            marked_advice=reply.get("marked_advice"),
            latency_ms=reply["latency_ms"],
            cost_usd=reply_cost(reply),
        )
        for (parsed, fired), reply in zip(verdicts, replies, strict=True)
    )
    return CaseResult(
        case_id=case["id"],
        expected_outcome=case["label"]["outcome"],
        expected_severity=case["label"]["severity"],
        outcome=outcome,
        severity=severity,
        attempts=attempts,
        reasons=reasons or tuple(reason for reason in logged if reason is not None),
    )


# --- the two modes: record once, replay as often as you like ----------------

DATASET = Path(__file__).parent / "dataset.jsonl"
RECORDINGS = Path(__file__).parent / "recordings.jsonl"


class RunFailed(Exception):
    """The run cannot be scored. The message says why and what to do about it."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def client_over(transport: httpx2.AsyncBaseTransport) -> anthropic.AsyncAnthropic:
    """Build an SDK client like `app.llm.get_client`'s, over `transport`."""
    settings = get_settings()
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.anthropic_timeout_seconds,
        max_retries=2,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=transport),
    )


async def record_cases(
    cases: list[dict[str, Any]], transport: httpx2.AsyncBaseTransport, recorded_at: str
) -> list[dict[str, Any]]:
    """Run every case against the real API; return the lines for recordings.jsonl.

    One line per reply, with `marked_advice` left null for a person to fill.
    Spends real money, bounded by the daily cap in config.

    Raises `RunFailed` unless every call got a complete reply. A timeout, an
    error status or a refused cap says something about the network or the
    account that day, not about the model, so nothing from such a run is kept.
    """
    recorder = Recorder(transport)
    client = client_over(recorder)
    ledger = SpendLedger(get_settings().daily_spend_cap_usd)
    lines: list[dict[str, Any]] = []
    for case in cases:
        recorder.exchanges.clear()
        result = await run_case(case, client, ledger, recorder.exchanges)
        # A reason in _VERDICTS means a reply did arrive: that is the model's
        # doing and is kept. Any other llm_ or spend_ reason means none did.
        failed = [x["status"] for x in recorder.exchanges if x["status"] != 200]
        failed += [
            reason
            for reason in result.reasons
            if reason.startswith(("llm_", "spend_")) and reason not in _VERDICTS
        ]
        if failed:
            raise RunFailed(f"{case['id']} did not get a complete reply: {failed}. Nothing saved.")
        for attempt, exchange in enumerate(recorder.exchanges, start=1):
            head = {"marked_advice": None, "case": case["id"], "attempt": attempt}
            lines.append({**head, "recorded_at": recorded_at, **exchange})
    return lines


async def replay_cases(
    cases: list[dict[str, Any]], recorded: list[dict[str, Any]]
) -> list[CaseResult]:
    """Run every case offline, answering Claude's part from `recorded`.

    Raises `RunFailed` if the recording no longer matches what the pipeline
    asks for: the prompt, the model, the dataset or the guard has changed
    since. Scoring it anyway would credit today's code with old replies.
    """
    replay = Replay(recorded)
    client = client_over(httpx2.MockTransport(replay))
    ledger = SpendLedger(math.inf)  # Replay spends nothing, so no cap applies.
    results = []
    for case in cases:
        replay.begin(case["id"])
        results.append(await run_case(case, client, ledger, replay.served))
    if replay.problems:
        listed = "\n  ".join(replay.problems)
        raise RunFailed(f"recordings.jsonl is out of date:\n  {listed}\nRun `make eval-record`.")
    return results


# --- the command line -------------------------------------------------------


def summary_lines(summary: Summary) -> list[str]:
    """Say each number as its counts. A rate with nothing under it stays "0 of 0"."""

    def ms(value: float | None) -> str:
        return "no calls" if value is None else f"{value:,.0f} ms"

    cost = summary.median_cost_usd
    return [
        "schema-valid   {} of {} replies".format(*summary.schema_valid),
        "severity       {} of {} delivered briefs match the label".format(
            *summary.severity_agreement
        ),
        "guard catch    {} of {} replies marked as advice".format(*summary.guard_catch),
        "fallback       {} of {} delivered alerts shipped unenriched".format(*summary.fallback),
        f"latency        p50 {ms(summary.latency_p50_ms)}, p95 {ms(summary.latency_p95_ms)}",
        "cost           " + ("no calls" if cost is None else f"median ${cost:.4f} per alert"),
    ]


def reply_lines(text: str) -> list[str]:
    """Split a reply into its headline and observations, or return it whole."""
    try:
        brief = json.loads(text)
        return [f"[{brief['severity']}] {brief['headline']}", *brief["observations"]]
    except (ValueError, KeyError, TypeError):  # Not a brief: show what arrived.
        return [text]


def unmarked_lines(cases: list[dict[str, Any]], recorded: list[dict[str, Any]]) -> list[str]:
    """Lay out each reply nobody has marked yet, so it can be read and marked.

    It does not say whether the guard fired. A marker who can see the guard's
    verdict drifts toward it, and the catch rate would then be the guard
    agreeing with itself.
    """
    conditions = {case["id"]: case["body"].get("condition", "") for case in cases}
    lines = []
    for reply in recorded:
        if reply["marked_advice"] is None:
            lines += ["", f"{reply['case']} attempt {reply['attempt']}"]
            lines += [f"  alert:  {conditions.get(reply['case'], '')!r}"]
            for block in reply["response"]["content"]:
                lines += [f"  reply:  {line}" for line in reply_lines(block.get("text", ""))]
    return lines


def main() -> None:
    cases = load_jsonl(DATASET)
    try:
        if "--record" in sys.argv[1:]:
            if RECORDINGS.exists() and any(
                line["marked_advice"] is not None for line in load_jsonl(RECORDINGS)
            ):
                raise RunFailed(f"{RECORDINGS.name} holds your marks. Delete it to record again.")
            recorded_at = datetime.now(UTC).isoformat(timespec="seconds")
            print(f"Recording {len(cases)} cases against the real API. About a minute.")
            lines = asyncio.run(record_cases(cases, httpx2.AsyncHTTPTransport(), recorded_at))
            text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
            RECORDINGS.write_text(text, encoding="utf-8")
            spent = sum(reply_cost(line) for line in lines)
            print(f"Saved {len(lines)} replies, ${spent:.4f} at list price. Next: make eval")
            return
        if not RECORDINGS.exists():
            raise RunFailed(f"No {RECORDINGS.name} yet. Run `make eval-record` first.")
        recorded = load_jsonl(RECORDINGS)
        summary = summarise(asyncio.run(replay_cases(cases, recorded)))
    except RunFailed as error:
        raise SystemExit(str(error)) from error
    print("\n".join(summary_lines(summary)))
    if summary.unmarked:
        print(
            f"\n{summary.unmarked} replies are not marked, so the guard catch rate is unfinished."
        )
        print(f"Set marked_advice to true or false on each line of eval/{RECORDINGS.name}:")
        print("\n".join(unmarked_lines(cases, recorded)))


if __name__ == "__main__":
    main()
