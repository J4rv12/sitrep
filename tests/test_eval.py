"""The eval dataset: 20 alerts, each with a label a person wrote.

A label says what the pipeline should do with the alert, written before any
model reply exists. `outcome` is one of:

    rejected    the webhook should answer 422 and deliver nothing
    unenriched  delivered without a brief, because no brief is possible
    enriched    delivered with a brief

`severity` is the brief's expected severity when the outcome is `enriched`,
and null otherwise: no brief, nothing to grade.

Severity is read from the case's `context` against the rubric in
app/prompts/brief_v1.md, never from numbers inside the alert text. That rubric
calls price "stretched" at roughly 10% from the 20MA and says nothing about
5-10%, so the labels settle it with one rule, fixed before any reply was
recorded:

    low     volume below its 20-day average
    high    volume 1.5x or more, and price within 10% of the 20MA either way
    medium  everything else, including 1.5x or more at 10% or further away

The `summarise` tests further down build their results by hand. The latencies
and costs in them are inputs for arithmetic, picked to be exact in binary so
sums can be compared without rounding. They are not recordings.
"""

import json
from pathlib import Path

import pytest

from eval.run_eval import Attempt, CaseResult, Rate, percentile, summarise

DATASET = Path(__file__).parent.parent / "eval" / "dataset.jsonl"
CASES = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]


def test_dataset_has_twenty_cases_across_three_categories() -> None:
    assert len({case["id"] for case in CASES}) == 20
    categories = [case["category"] for case in CASES]
    assert {c: categories.count(c) for c in set(categories)} == {
        "normal": 8,
        "edge": 6,
        "adversarial": 6,
    }


def test_every_case_is_labelled() -> None:
    for case in CASES:
        outcome, severity = case["label"]["outcome"], case["label"]["severity"]
        assert outcome in ("rejected", "unenriched", "enriched"), f"{case['id']}: outcome"
        if outcome == "enriched":
            assert severity in ("low", "medium", "high"), f"{case['id']}: severity"
            # Invariant 4: without market numbers Claude is never called.
            assert case["context"] is not None, f"{case['id']}: a brief needs context"
        else:
            assert severity is None, f"{case['id']}: no brief, so no severity"


# --- percentile -------------------------------------------------------------


def test_percentile_returns_a_value_that_occurred() -> None:
    assert percentile([400.0, 100.0, 300.0, 200.0], 50) == 200.0
    assert percentile([float(n) for n in range(1, 21)], 95) == 19.0
    assert percentile([float(n) for n in range(1, 17)], 95) == 16.0, "with 16, p95 is the max"
    assert percentile([7.0], 95) == 7.0


def test_percentile_of_nothing_raises() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


# --- summarise: yours -------------------------------------------------------


def reply(
    *,
    valid: bool = True,
    fired: bool = False,
    marked: bool | None = False,
    ms: float = 1000.0,
    usd: float = 0.25,
) -> Attempt:
    """One reply. By default: parsed, guard silent, marked not-advice."""
    return Attempt(
        schema_valid=valid, guard_fired=fired, marked_advice=marked, latency_ms=ms, cost_usd=usd
    )


def alert(
    *attempts: Attempt,
    outcome: str = "enriched",
    severity: str | None = "high",
    expected_outcome: str = "enriched",
    expected_severity: str | None = "high",
) -> CaseResult:
    """One case. By default: expected a high brief and delivered one."""
    return CaseResult(
        case_id="case",
        expected_outcome=expected_outcome,
        expected_severity=expected_severity,
        outcome=outcome,
        severity=severity,
        attempts=attempts,
    )


def test_schema_valid_rate_counts_replies_not_alerts() -> None:
    retried = alert(reply(valid=False), reply())
    first_time = alert(reply())

    assert summarise([retried, first_time]).schema_valid == Rate(2, 3)


def test_severity_agreement_grades_only_delivered_briefs() -> None:
    agrees = alert(reply(), severity="high", expected_severity="high")
    disagrees = alert(reply(), severity="medium", expected_severity="high")
    no_brief = alert(reply(valid=False), reply(valid=False), outcome="unenriched", severity=None)

    assert summarise([agrees, disagrees, no_brief]).severity_agreement == Rate(1, 2)


def test_guard_catch_rate_is_over_replies_marked_as_advice() -> None:
    caught = reply(fired=True, marked=True)
    missed = reply(fired=False, marked=True)
    false_alarm = reply(fired=True, marked=False)
    clean = reply(fired=False, marked=False)

    summary = summarise([alert(caught, missed), alert(false_alarm, clean)])

    assert summary.guard_catch == Rate(1, 2)


def test_no_advice_is_zero_of_zero() -> None:
    summary = summarise([alert(reply()), alert(reply(fired=True))])

    assert summary.guard_catch == Rate(0, 0)


def test_advice_in_a_reply_that_never_parsed_is_not_the_guards_to_catch() -> None:
    summary = summarise([alert(reply(valid=False, marked=True), reply())])

    assert summary.guard_catch == Rate(0, 0)


def test_unmarked_replies_are_counted_and_kept_out_of_the_catch_rate() -> None:
    summary = summarise(
        [alert(reply(marked=None, fired=True), reply(valid=False, marked=None)), alert(reply())]
    )

    assert summary.unmarked == 2
    assert summary.guard_catch == Rate(0, 0)


def test_fallback_rate_is_over_delivered_alerts() -> None:
    results = [
        alert(reply()),
        alert(outcome="unenriched", severity=None, expected_outcome="unenriched"),
        alert(reply(fired=True), reply(fired=True), outcome="unenriched", severity=None),
        alert(outcome="rejected", severity=None, expected_outcome="rejected"),
        alert(outcome="crashed", severity=None, expected_outcome="rejected"),
    ]

    assert summarise(results).fallback == Rate(2, 3)


def test_latency_and_cost_are_per_alert_with_retries_added_in() -> None:
    retried = alert(reply(ms=1000.0, usd=0.25), reply(ms=2000.0, usd=0.25))
    quick = alert(reply(ms=500.0, usd=0.125))
    middle = alert(reply(ms=1500.0, usd=0.25))
    never_called = alert(outcome="unenriched", severity=None)

    summary = summarise([retried, quick, middle, never_called])

    # Per alert: 3000, 500, 1500 ms and 0.5, 0.125, 0.25 USD.
    assert summary.latency_p50_ms == 1500.0
    assert summary.latency_p95_ms == 3000.0
    assert summary.median_cost_usd == 0.25


def test_a_run_with_no_claude_calls_has_no_latency_or_cost() -> None:
    for results in ([], [alert(outcome="rejected", severity=None)]):
        summary = summarise(results)

        assert summary.schema_valid == Rate(0, 0)
        assert summary.severity_agreement == Rate(0, 0)
        assert summary.unmarked == 0
        assert summary.latency_p50_ms is None
        assert summary.latency_p95_ms is None
        assert summary.median_cost_usd is None
