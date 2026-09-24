"""Every environment variable this service reads, typed and in one place.

`os.getenv` anywhere else in `app/` is a review rejection. The point is that
one file tells you everything the service needs to boot.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and `.env`.

    Secrets have no defaults: a missing one must stop the process at boot
    rather than surface as a 500 on the first real alert.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Render injects PORT and friends; not our business.
        # A rejected value is otherwise quoted in the error, and the error goes
        # to Render's deploy log. A real token one character too short would
        # be published there in full.
        hide_input_in_errors=True,
    )

    # --- Secrets. No defaults, ever. ---
    # 32 rejects `replace-me`, the placeholder anyone can read in the public
    # .env.example, and any token short enough to guess. The generator
    # .env.example suggests produces 43. With no minimum, an empty value
    # boots, and then a request carrying `"token": ""` authenticates.
    webhook_token: str = Field(min_length=32)
    # Anthropic checks the key on the first call. This only stops an empty
    # line from booting a service whose every brief then fails with a 401.
    anthropic_api_key: str = Field(min_length=1)

    # An empty line in .env (`TELEGRAM_BOT_TOKEN=`) reads as "", which is a
    # valid str. Without min_length the service would boot and then fail
    # every delivery; with it, the deploy fails instead.
    telegram_bot_token: str = Field(min_length=1)
    telegram_chat_id: str = Field(min_length=1)

    # --- Operational settings. Not secret, so defaults are safe here. ---
    anthropic_model: str = "claude-haiku-4-5-20251001"
    # The pinned model's list price. Change these together with the model, or
    # the spend ledger silently counts the wrong dollars.
    anthropic_input_usd_per_mtok: float = 1.00
    anthropic_output_usd_per_mtok: float = 5.00
    # Upper bound on one call's input, for the spend ledger's worst case. brief_v1
    # measured 1,223 tokens on 2026-09-15 with a 16-character condition; the
    # longest alert AlertPayload allows adds a few hundred. Re-measure whenever the
    # prompt changes; tests/test_llm.py fails if a recording exceeds this.
    anthropic_max_input_tokens: int = 2000
    # A full brief is ~300 tokens of JSON. A truncated one is invalid JSON and
    # costs a retry, so leave headroom rather than trimming this to fit.
    anthropic_max_tokens: int = 1024
    # The SDK's default is ten minutes. The SDK also retries a timeout twice,
    # so one attempt can take up to three times this.
    anthropic_timeout_seconds: float = 20.0
    # Invariant 6 is one call per alert; a second attempt is allowed only for a
    # response that fails validation or the advice guard. Both are billed.
    llm_max_attempts: int = 2
    # Per process, so a restart resets it. The hard cap is the monthly spend
    # limit on SitRep's workspace in the Anthropic Console.
    daily_spend_cap_usd: float = 1.00
    dedupe_ttl_seconds: int = 300  # Invariant 5: a 5-minute dedupe window.
    # Binance's market-data-only host: public endpoints, no auth, which is all
    # we call. Also reachable where api.binance.com is blocked at the ISP.
    binance_base_url: str = "https://data-api.binance.vision"
    telegram_base_url: str = "https://api.telegram.org"
    http_timeout_seconds: float = 5.0  # Binance and Telegram both.
    demo_rate_limit_per_hour: int = 20
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, built once and cached.

    Raises `pydantic.ValidationError` on the first call if a required
    variable is missing or cannot be coerced to its declared type. Call it
    during app startup so that failure happens at boot, not mid-request.

    Tests override configuration by setting environment variables and then
    calling `get_settings.cache_clear()`.
    """
    return Settings()
