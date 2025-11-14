"""Settings for {{cookiecutter.component_name}} component.

This module defines all configuration settings using Pydantic Settings,
allowing configuration via environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for {{cookiecutter.component_name}}.

    All settings can be overridden using environment variables with the
    prefix '{{cookiecutter.component_name.replace('-', '_').upper()}}_'.

    Attributes:
        host: API server host.
        port: API server port.
        log_level: Logging level.
    """

    model_config = SettingsConfigDict(
        env_prefix="{{cookiecutter.component_name.replace('-', '_').upper()}}_",
        case_sensitive=False,
    )

    # API Settings
    host: str = "0.0.0.0"
    port: int = {{cookiecutter.component_port}}
    log_level: str = "INFO"

    # Component-specific settings
    # TODO: Add component-specific configuration here


# Global settings instance
settings = Settings()
