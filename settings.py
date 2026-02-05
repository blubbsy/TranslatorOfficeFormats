"""
Application settings loaded from environment variables.
Uses Pydantic Settings for type-safe configuration with validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from .env file."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # LLM Connection
    llm_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL for OpenAI-compatible LLM API"
    )
    llm_api_key: str = Field(
        default="lm-studio",
        description="API key for LLM service"
    )
    llm_model: str = Field(
        default="llama-3-8b-instruct",
        description="Model identifier to use for translation"
    )
    llm_context_window: int = Field(
        default=4096,
        description="Max context window size (tokens) of the LLM"
    )
    
    # App Behavior
    target_language: str = Field(
        default="English",
        description="Target language for translations"
    )
    ocr_source_languages: str = Field(
        default="en,de,zh-cn",
        description="Comma-separated list of source languages for OCR (e.g., en,de,fr,zh-cn)"
    )
    translate_images: bool = Field(
        default=True,
        description="Whether to OCR and translate text in images"
    )
    image_translation_method: str = Field(
        default="ocr",
        description="Method for image translation: 'ocr' or 'vlm'"
    )
    
    # Advanced Features
    preserve_formatting: bool = Field(
        default=False,
        description="Preserve bold/italic formatting (more complex, may be less reliable)"
    )
    context_window_size: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Number of previous paragraphs to include as context"
    )
    glossary_file: Optional[str] = Field(
        default=None,
        description="Path to glossary JSON file for consistent translations"
    )
    
    # System
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for LLM calls"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper_v
    
    @field_validator("glossary_file")
    @classmethod
    def validate_glossary_file(cls, v: Optional[str]) -> Optional[str]:
        if v and v.strip():
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Glossary file not found: {v}")
            return str(path.resolve())
        return None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience singleton
settings = get_settings()
