"""Application configuration, sourced only from the environment."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    `anthropic_api_key` is read from the `ANTHROPIC_API_KEY` environment
    variable (or a local `.env` file, for development convenience — see
    `.env.example`). It is never hardcoded and has no default: if it is
    missing, instantiating `Settings` raises a `ValidationError` instead of
    silently falling back to an empty or placeholder key.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr = Field(
        description="API key for the Anthropic SDK, read from ANTHROPIC_API_KEY."
    )


settings = Settings()
