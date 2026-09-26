"""Configuration settings for Stremio Arabic Subtitles Addon (Vercel & Local Ready)."""

import os
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Microservice environment and performance settings."""

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 7000
    BASE_URL: str | None = None  # e.g., http://192.168.1.50:7000
    ADDON_BASE_URL: str | None = None
    HOST_IP: str | None = None
    LAN_IP: str | None = None

    # Upstream Provider API Keys
    SUBDL_API_KEY: str = ""
    SUBSOURCE_API_KEY: str = ""
    OPENSUBTITLES_API_KEY: str = ""

    # Keyless scraper providers (no API key required)
    ENABLE_YIFYSUBTITLES: bool = True
    ENABLE_SUBTITLECAT: bool = True

    # Disk Cache limits (/tmp for serverless environments)
    CACHE_DIR: str = os.getenv("CACHE_DIR", "/tmp/subs_cache" if os.getenv("VERCEL") else str(Path.cwd() / "subs_cache"))
    CACHE_MAX_BYTES: int = 1 * 1024 * 1024 * 1024  # 1 GB
    CACHE_MAX_FILES: int = 500

    # Networking & Resource Constraints (<100MB RAM, strict 6s timeout)
    UPSTREAM_TIMEOUT: float = 6.0
    MAX_KEEP_ALIVE_CONNECTIONS: int = 20
    MAX_CONNECTIONS: int = 50

    # Diagnostic / Observability
    NINJASUBS_DEBUG_RANKING: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator(
        "PORT",
        "CACHE_MAX_BYTES",
        "CACHE_MAX_FILES",
        "MAX_KEEP_ALIVE_CONNECTIONS",
        "MAX_CONNECTIONS",
        mode="before",
    )
    @classmethod
    def parse_empty_int(cls, v: Any, info) -> Any:
        if v is None or v == "":
            defaults = {
                "PORT": 7000,
                "CACHE_MAX_BYTES": 1073741824,
                "CACHE_MAX_FILES": 500,
                "MAX_KEEP_ALIVE_CONNECTIONS": 20,
                "MAX_CONNECTIONS": 50,
            }
            return defaults.get(info.field_name, 0)
        return int(v)

    @field_validator(
        "ENABLE_YIFYSUBTITLES",
        "ENABLE_SUBTITLECAT",
        "NINJASUBS_DEBUG_RANKING",
        mode="before",
    )
    @classmethod
    def parse_empty_bool(cls, v: Any, info) -> bool:
        if v is None or v == "":
            defaults = {
                "ENABLE_YIFYSUBTITLES": True,
                "ENABLE_SUBTITLECAT": True,
                "NINJASUBS_DEBUG_RANKING": False,
            }
            return defaults.get(info.field_name, False)
        if isinstance(v, str):
            return v.strip().lower() not in ("0", "false", "no", "off")
        return bool(v)

    @field_validator("UPSTREAM_TIMEOUT", mode="before")
    @classmethod
    def parse_empty_float(cls, v: Any) -> float:
        if v is None or v == "":
            return 6.0
        return float(v)


settings = Settings()
