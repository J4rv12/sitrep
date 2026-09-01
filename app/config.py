"""Every environment variable this service reads, typed and in one place.

`os.getenv` anywhere else in `app/` is a review rejection. The point is that
one file tells you everything the service needs to boot.
"""

from functools import lru_cache

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
    )

    # --- Secrets. No defaults, ever. ---
    webhook_token: str
    anthropic_api_key: str

    # Required from Phase 4, when Telegram delivery lands. Optional until then
    # so Phase 0 can deploy before you have a bot token from @BotFather.
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    # --- Operational settings. Not secret, so defaults are safe here. ---
    anthropic_model: str = "claude-haiku-4-5-20251001"
    daily_spend_cap_usd: float = 1.00
    dedupe_ttl_seconds: int = 300  # Invariant 5: a 5-minute dedupe window.
    binance_base_url: str = "https://api.binance.com"
    http_timeout_seconds: float = 5.0
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
