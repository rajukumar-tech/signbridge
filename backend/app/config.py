from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./signbridge.db"
    jwt_secret: str = "dev-insecure-secret-change-me-please-32b+"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost:8000"
    reviewer_emails: str = ""
    seed_on_startup: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def reviewer_email_set(self) -> set[str]:
        return {e.strip().lower() for e in self.reviewer_emails.split(",") if e.strip()}

    @property
    def sqlalchemy_url(self) -> str:
        url = self.database_url
        # Normalise plain postgres URLs to the psycopg (v3) driver.
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
