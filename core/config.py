from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    `groq_api_key` is read from the `GROQ_API_KEY` environment variable (or
    a local `.env` file, for development convenience — see `.env.example`).
    It is never hardcoded and has no default: if it is missing, instantiating
    `Settings` raises a `ValidationError` instead of silently falling back to
    an empty or placeholder key.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: SecretStr = Field(
        description="API key for the Groq SDK, read from GROQ_API_KEY."
    )


settings = Settings()
