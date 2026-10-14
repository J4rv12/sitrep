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

The record and replay tests at the bottom use the one real Claude reply the
suite already has, the fixture tests/test_llm.py describes. They record it
offline, through the `Recorder` over a transport that serves the fixture, so
its latency there is not a real one and only its shape is checked.
"""

import asyncio
import json
from pathlib import Path

import anthropic
import httpx2
import pytest

from app import llm
from app.enrich import compute_context
from app.schemas import AlertPayload, SignalBrief
from eval import run_eval
from eval.run_eval import (
    Attempt,
    CaseResult,
    Rate,
    Recorder,
    Replay,
    RunFailed,
    percentile,
    record_cases,
    render_results,
    replay_cases,
    run_case,
    summarise,
)

FIXTURES = Path(__file__).parent / "fixtures"
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


# --- record and replay ------------------------------------------------------

REPLY = json.loads((FIXTURES / "anthropic_brief_btcusdt_low_volume.json").read_bytes())
_KLINES = json.loads((FIXTURES / "binance_klines_btcusdt_1d.json").read_bytes())
CONTEXT = compute_context([float(r[4]) for r in _KLINES], [float(r[5]) for r in _KLINES])
ALERT = AlertPayload(
    symbol="BTCUSDT", timeframe="1h", condition="close below 20MA", bar_time="2026-09-10T10:00:00Z"
)


async def brief_over(
    transport: httpx2.AsyncBaseTransport, alert: AlertPayload = ALERT
) -> SignalBrief | None:
    """Run the real `generate_brief` on a real SDK client over `transport`."""
    client = anthropic.AsyncAnthropic(
        api_key="sk-ant-test-not-a-real-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=transport),
    )
    return await llm.generate_brief(
        alert, CONTEXT, "test-alert", client=client, ledger=llm.SpendLedger(cap_usd=1.0)
    )


async def recorded_lines() -> list[dict[str, object]]:
    """One exchange as recordings.jsonl would hold it, for a case called c1."""
    recorder = Recorder(httpx2.MockTransport(lambda request: httpx2.Response(200, json=REPLY)))
    await brief_over(recorder)
    return [{"case": "c1", **exchange} for exchange in recorder.exchanges]


async def test_a_recorded_reply_replays_into_the_same_brief() -> None:
    lines = await recorded_lines()
    (line,) = lines
    assert (line["status"], line["response"]) == (200, REPLY)
    assert len(str(line["request_sha256"])) == 64

    replay = Replay(lines)
    replay.begin("c1")
    brief = await brief_over(httpx2.MockTransport(replay))

    assert brief == SignalBrief.model_validate_json(REPLY["content"][0]["text"])
    assert replay.served == lines
    assert replay.problems == []


async def test_replay_refuses_a_request_that_changed_since_recording() -> None:
    """A different condition stands in for an edited prompt: both change the request."""
    replay = Replay(await recorded_lines())
    replay.begin("c1")

    brief = await brief_over(
        httpx2.MockTransport(replay), ALERT.model_copy(update={"condition": "close above 20MA"})
    )

    assert brief is None
    assert replay.served == []
    assert replay.problems == ["c1: the request has changed since it was recorded"]


async def test_replay_refuses_to_answer_more_often_than_was_recorded() -> None:
    replay = Replay(await recorded_lines())
    replay.begin("c1")
    await brief_over(httpx2.MockTransport(replay))

    again = await brief_over(httpx2.MockTransport(replay))

    assert again is None
    assert replay.problems == ["c1: asked for a reply that was never recorded"]


# --- one case through the pipeline ------------------------------------------

REPLY_TEXT = REPLY["content"][0]["text"]
# The recorded reply with an advice-shaped headline swapped in, as in
# tests/test_llm.py. The swapped text is written for the test, not model output.
ADVICE_REPLY = {
    **REPLY,
    "content": [
        {
            "type": "text",
            "text": json.dumps({**json.loads(REPLY_TEXT), "headline": "Strong buy setup."}),
        }
    ],
}
CASE = {
    "id": "c1",
    "category": "normal",
    "note": "",
    "body": ALERT.model_dump(mode="json"),
    "context": CONTEXT.model_dump(),
    "label": {"outcome": "enriched", "severity": "low"},
}


async def run_over(case: dict, *bodies: dict) -> CaseResult:
    """Run `case` on a client whose transport records `bodies`, served in order."""
    waiting = iter(bodies)
    recorder = Recorder(
        httpx2.MockTransport(lambda request: httpx2.Response(200, json=next(waiting)))
    )
    client = anthropic.AsyncAnthropic(
        api_key="sk-ant-test-not-a-real-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=recorder),
    )
    return await run_case(case, client, llm.SpendLedger(cap_usd=1.0), recorder.exchanges)


async def test_a_clean_reply_is_one_attempt_and_an_enriched_outcome() -> None:
    result = await run_over(CASE, REPLY)

    assert (result.outcome, result.severity, result.reasons) == ("enriched", "low", ())
    (attempt,) = result.attempts
    assert (attempt.schema_valid, attempt.guard_fired, attempt.marked_advice) == (True, False, None)
    usage = REPLY["usage"]
    assert attempt.cost_usd == llm.cost_usd(usage["input_tokens"], usage["output_tokens"])


async def test_a_reply_the_guard_rejects_is_kept_as_its_own_attempt() -> None:
    result = await run_over(CASE, ADVICE_REPLY, REPLY)

    assert (result.outcome, result.reasons) == ("enriched", ("advice_detected",))
    assert [(a.schema_valid, a.guard_fired) for a in result.attempts] == [
        (True, True),
        (True, False),
    ]


async def test_advice_twice_ships_unenriched() -> None:
    result = await run_over(CASE, ADVICE_REPLY, ADVICE_REPLY)

    assert (result.outcome, result.severity) == ("unenriched", None)
    assert result.reasons == ("advice_detected", "advice_detected")
    assert len(result.attempts) == 2


async def test_a_reply_that_breaks_the_schema_is_an_invalid_attempt() -> None:
    broken = {**REPLY, "content": [{"type": "text", "text": REPLY_TEXT[:40]}]}

    result = await run_over(CASE, broken, REPLY)

    assert [a.schema_valid for a in result.attempts] == [False, True]
    assert result.reasons == ("llm_invalid_output",)


async def test_a_case_without_market_numbers_never_calls_claude() -> None:
    result = await run_over({**CASE, "context": None})

    assert (result.outcome, result.attempts) == ("unenriched", ())
    assert result.reasons == ("no_market_context",)


async def test_a_body_that_fails_validation_is_rejected() -> None:
    body = {k: v for k, v in CASE["body"].items() if k != "condition"}

    result = await run_over({**CASE, "body": body})

    assert (result.outcome, result.attempts, result.reasons) == ("rejected", (), ())


async def test_an_exception_before_the_pipeline_is_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(payload: AlertPayload) -> str:
        raise RuntimeError("simulated")

    monkeypatch.setattr("eval.run_eval.compute_alert_id", boom)

    result = await run_over(CASE)

    assert (result.outcome, result.reasons) == ("crashed", ("RuntimeError",))


# --- the two modes ----------------------------------------------------------

NO_CONTEXT = {
    **CASE,
    "id": "c2",
    "context": None,
    "label": {"outcome": "unenriched", "severity": None},
}


def serving(*bodies: dict, status: int = 200) -> httpx2.MockTransport:
    waiting = iter(bodies)
    return httpx2.MockTransport(lambda request: httpx2.Response(status, json=next(waiting)))


async def test_what_record_saves_replay_scores_the_same_way() -> None:
    cases = [CASE, NO_CONTEXT]

    lines = await record_cases(cases, serving(ADVICE_REPLY, REPLY), "2026-10-10T00:00:00+00:00")

    assert [(line["case"], line["attempt"], line["marked_advice"]) for line in lines] == [
        ("c1", 1, None),
        ("c1", 2, None),
    ]
    results = await replay_cases(cases, lines)
    assert [(r.case_id, r.outcome, len(r.attempts)) for r in results] == [
        ("c1", "enriched", 2),
        ("c2", "unenriched", 0),
    ]
    assert [a.guard_fired for a in results[0].attempts] == [True, False]


async def test_record_keeps_nothing_from_a_run_the_api_refused() -> None:
    refusal = {"type": "error", "error": {"type": "invalid_request_error", "message": "simulated"}}

    with pytest.raises(RunFailed, match="c1 did not get a complete reply"):
        await record_cases([CASE], serving(refusal, status=400), "2026-10-10T00:00:00+00:00")


async def test_replay_stops_on_a_recording_that_no_longer_matches() -> None:
    lines = await record_cases([CASE], serving(REPLY), "2026-10-10T00:00:00+00:00")
    edited = {**CASE, "body": {**CASE["body"], "condition": "close above 20MA"}}

    with pytest.raises(RunFailed, match="out of date"):
        await replay_cases([edited], lines)


# --- the command line -------------------------------------------------------


async def write_recording(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module at a one-case dataset and a recording of it under `path`."""
    lines = await record_cases([CASE], serving(REPLY), "2026-10-10T00:00:00+00:00")
    (path / "dataset.jsonl").write_text(json.dumps(CASE) + "\n", encoding="utf-8")
    (path / "recordings.jsonl").write_text(json.dumps(lines[0]) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "DATASET", path / "dataset.jsonl")
    monkeypatch.setattr(run_eval, "RECORDINGS", path / "recordings.jsonl")
    monkeypatch.setattr(run_eval, "RESULTS", path / "results.md")
    monkeypatch.setattr("sys.argv", ["run_eval"])


async def test_replay_prints_the_score_and_the_replies_still_to_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    await write_recording(tmp_path, monkeypatch)

    # `main` starts its own event loop, which cannot be done from inside this one.
    await asyncio.to_thread(run_eval.main)

    printed = capsys.readouterr().out
    assert "schema-valid   1 of 1 replies" in printed
    assert "guard catch    0 of 0 replies marked as advice" in printed
    assert "1 replies are not marked" in printed
    assert "c1 attempt 1" in printed and "close below 20MA" in printed
    assert "| c1 | " in (tmp_path / "results.md").read_text(encoding="utf-8")


async def test_replay_without_a_recording_says_what_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await write_recording(tmp_path, monkeypatch)
    (tmp_path / "recordings.jsonl").unlink()

    with pytest.raises(SystemExit, match="make eval-record"):
        run_eval.main()


# --- the committed recording and results ------------------------------------
#
# These three run the real dataset against the real recording, offline. They
# are what turns "re-run the eval when the prompt changes" from a habit into
# a failing build.


@pytest.fixture(scope="module")
def committed() -> tuple[list[dict], list[dict], list[CaseResult]]:
    recorded = run_eval.load_jsonl(run_eval.RECORDINGS)
    return CASES, recorded, asyncio.run(replay_cases(CASES, recorded))


def test_the_recording_still_answers_what_the_pipeline_asks(
    committed: tuple[list[dict], list[dict], list[CaseResult]],
) -> None:
    """Fails with `RunFailed` if the prompt, model, dataset or guard changed without
    a new recording."""
    _, recorded, results = committed

    assert sum(len(result.attempts) for result in results) == len(recorded)


def test_no_reply_marked_as_advice_was_delivered(
    committed: tuple[list[dict], list[dict], list[CaseResult]],
) -> None:
    """Invariant 2, held against real model output. Every reply has been read."""
    summary = summarise(committed[2])

    assert summary.unmarked == 0
    assert summary.guard_catch.hits == summary.guard_catch.total


def test_results_md_is_what_a_fresh_replay_writes(
    committed: tuple[list[dict], list[dict], list[CaseResult]],
) -> None:
    cases, recorded, results = committed

    fresh = render_results(cases, recorded, results, summarise(results))

    assert run_eval.RESULTS.read_text(encoding="utf-8") == fresh, "run `make eval` and commit"
