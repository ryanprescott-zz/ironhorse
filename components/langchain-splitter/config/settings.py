"""Settings for langchain-splitter component.

This module defines all configuration settings using Pydantic Settings,
allowing configuration via environment variables.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for langchain-splitter.

    All settings can be overridden using environment variables with the
    prefix 'LANGCHAIN_SPLITTER_'.

    Attributes:
        host: API server host.
        port: API server port.
        log_level: Logging level.
        default_chunk_size: Default chunk size in characters.
        default_chunk_overlap: Default overlap between chunks.
        default_separators: Default list of separators for splitting.
    """

    model_config = SettingsConfigDict(
        env_prefix="LANGCHAIN_SPLITTER_",
        case_sensitive=False,
    )

    # API Settings
    host: str = "0.0.0.0"
    port: int = 26001
    log_level: str = "INFO"

    # Splitter Settings
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_separators: List[str] = ["\n\n", "\n", " ", ""]
    default_keep_separator: bool = True
    default_strip_whitespace: bool = True


# Global settings instance
settings = Settings()
