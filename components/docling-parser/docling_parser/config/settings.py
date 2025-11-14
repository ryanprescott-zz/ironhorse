"""Settings for docling-parser component.

This module defines all configuration settings using Pydantic Settings,
allowing configuration via environment variables.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for docling-parser.

    All settings can be overridden using environment variables with the
    prefix 'DOCLING_PARSER_'.

    Attributes:
        host: API server host.
        port: API server port.
        log_level: Logging level.
        supported_formats: List of supported file formats.
        max_file_size_mb: Maximum file size in MB.
        extract_tables: Whether to extract tables from documents.
        extract_images: Whether to extract images from documents.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_PARSER_",
        case_sensitive=False,
    )

    # API Settings
    host: str = "0.0.0.0"
    port: int = 26000
    log_level: str = "INFO"

    # Parser Settings
    supported_formats: List[str] = ["pdf", "docx", "html", "xlsx", "pptx"]
    max_file_size_mb: int = 100
    extract_tables: bool = True
    extract_images: bool = False  # For future use


# Global settings instance
settings = Settings()
