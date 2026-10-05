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
"""

import json
from pathlib import Path

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
