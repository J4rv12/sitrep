"""Test-wide setup. Lives at the repo root for two reasons.

First, pytest prepends this directory to `sys.path`, so `import app` resolves
without packaging the project.

Second, `app.main` calls `get_settings()` at import time, so required secrets
must exist in the environment *before* any test module imports it. CI has no
`.env` file; without this, every test would fail at collection.

These values are set unconditionally, not via `setdefault`, so a developer's
real `.env` can never leak into a test run.
"""

import os

os.environ["WEBHOOK_TOKEN"] = "test-webhook-token-at-least-32-chars"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-not-a-real-key"
os.environ["TELEGRAM_BOT_TOKEN"] = "123456789:test-telegram-token"
os.environ["TELEGRAM_CHAT_ID"] = "-1000000000001"


class FakeClock:
    """A monotonic clock you control, for anything with a TTL.

    Shared by tests/test_dedupe.py and tests/test_webhook.py so the two
    cannot drift apart on what "time" means.

    Starts at an arbitrary non-zero value so an implementation that treats
    0.0 as "never seen" fails rather than passing by accident.
    """

    def __init__(self, now: float = 10_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
