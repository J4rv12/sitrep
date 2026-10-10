"""The eval: what one run produced, and the six numbers drawn from it.

`CaseResult` is what the pipeline did with one dataset case. `summarise`
turns a list of them into a `Summary`. Both are plain data with no I/O, so
the arithmetic can be tested without a recording.
"""

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

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
