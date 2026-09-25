"""Boot-time checks on the secrets.

Each test builds a fresh `Settings` rather than going through `get_settings`,
whose cached instance the rest of the suite shares. `_env_file=None` keeps a
developer's real `.env` out of it; the environment conftest.py set up supplies
every other required value.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def build(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


@pytest.mark.parametrize(
    ("token", "case"),
    [
        ("", "an empty line in .env or Render"),
        ("replace-me", "the placeholder from the public .env.example"),
        ("x" * 31, "one character short"),
    ],
)
def test_short_webhook_token_fails_at_boot(
    monkeypatch: pytest.MonkeyPatch, token: str, case: str
) -> None:
    with pytest.raises(ValidationError):
        build(monkeypatch, WEBHOOK_TOKEN=token)


def test_webhook_token_of_32_characters_boots(monkeypatch: pytest.MonkeyPatch) -> None:
    assert build(monkeypatch, WEBHOOK_TOKEN="x" * 32).webhook_token == "x" * 32


def test_empty_anthropic_key_fails_at_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        build(monkeypatch, ANTHROPIC_API_KEY="")


def test_demo_is_off_unless_a_deployment_opts_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_ENABLED", raising=False)

    assert build(monkeypatch).demo_enabled is False


def test_rejected_secret_is_not_quoted_in_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real-looking token, one character short. The error is printed to
    # Render's deploy log, so the token must not appear in it.
    secret = "Zq8vN3kLr5TtW1yXb7Hc2Mf9Gd4Js6A"
    assert len(secret) == 31

    with pytest.raises(ValidationError) as excinfo:
        build(monkeypatch, WEBHOOK_TOKEN=secret)

    assert "webhook_token" in str(excinfo.value)
    assert secret not in str(excinfo.value)
