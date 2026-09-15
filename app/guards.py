"""Invariant 2, enforced on the output: no advice reaches Telegram.

The prompt asks the model not to advise. That lowers the odds and guarantees
nothing, because the alert's `condition` is written by whoever set up the
alert and can ask for exactly the opposite. So every brief passes through
here after it parses and before it is formatted. A brief that trips the guard
is discarded, and the alert ships unenriched.

Deterministic on purpose. A second model asked "is this advice?" would be a
second paid call per alert (invariant 6), could give two answers for the same
text, and would read the same attacker-written condition the first one did.
"""

import re

from app.schemas import SignalBrief

PATTERN = re.compile(
    r"\b(buy|sell|hold|short(?!-)|enter|entering|entry|exit\w*|target\w*|long(?!-)|stop[- ]?loss"
    r"|take[- ]?profit|should|consider|recommend\w*|wait for|look to)\b",
    re.IGNORECASE,
)


def find_advice(brief: SignalBrief) -> str | None:
    """Return the first advice-shaped phrase in `brief`, or None if it has none.

    Scans `headline` and every item in `observations`. `severity` is a
    Literal of three fixed words and cannot carry prose, so it is not checked.

    Returns the matched text as it appears in the brief, not a bool, so a
    rejected brief or a failing eval case shows exactly what tripped it.

    Advice is wording that tells the reader what to do or where price will
    go: buy, sell, hold, short, long, enter, entry, exit, target, stop loss,
    take profit, should, consider, recommend, wait for, look to. Stop loss
    and take profit match joined by a space, a hyphen, or nothing.

    Market description borrows the same stems: "buying interest", "sellers",
    "holding below the average", "volume has stopped expanding", "short-term".
    So a word form is listed only if every way a brief would use it is
    advice. "Exiting", "targets" and "recommended" qualify; "buying",
    "sellers" and "holding" do not. A regex word boundary sits on either side
    of a hyphen, which is why "short" refuses a hyphen after it.

    The list is a floor, not a definition. When a miss turns up — in review,
    in the eval, or in production logs — it becomes a case in
    tests/test_guards.py before the pattern changes.

    When in doubt, catch it. A false positive costs one alert delivered
    without its brief — visible, and cheap. A false negative ships financial
    advice, the one thing SitRep exists not to do.

    Pure: no I/O, no config, no logging. The caller logs the reason code.
    Never raises for any valid `SignalBrief`.
    """
    for text in [brief.headline, *brief.observations]:
        m = PATTERN.search(text)
        if m:
            return m.group()

    return None
