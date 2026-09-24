"""Token verification contract.

conftest.py sets WEBHOOK_TOKEN to "test-webhook-token-at-least-32-chars" before
anything imports app.config, so that is the secret under test throughout this
file.

The assertions use `is True` / `is False` rather than `assert token_is_valid(...)`.
A truthy return is not good enough here: this value decides whether a request
is authenticated, and "returns something truthy" is a weaker contract than
"returns a bool" when the next person refactors it.
"""

import pytest

from app.security import token_is_valid

VALID = "test-webhook-token-at-least-32-chars"


def test_correct_token_is_accepted() -> None:
    assert token_is_valid(VALID) is True


@pytest.mark.parametrize(
    ("supplied", "case"),
    [
        (VALID[:-1] + "S", "final character differs"),
        (VALID[:-1], "a correct prefix, one character short"),
        (VALID + "-extra", "correct token plus a suffix"),
        ("", "empty string"),
        (None, "no token key in the body at all"),
        (12345, "a JSON number rather than a string"),
        ([VALID], "the correct token, wrapped in a list"),
        (VALID.replace("e", "ë", 1), "non-ASCII"),
    ],
)
def test_bad_tokens_are_rejected(supplied: object, case: str) -> None:
    assert token_is_valid(supplied) is False, case
