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

os.environ["WEBHOOK_TOKEN"] = "test-webhook-token"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-not-a-real-key"
