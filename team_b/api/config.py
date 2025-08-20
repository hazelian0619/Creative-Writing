"""Configuration management for cognitive state backend."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # MongoDB Configuration
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URL")
    database_name: str = Field(default="cognitive_state_db", description="Database name")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    api_reload: bool = Field(default=True, description="Enable auto-reload in development")
    api_workers: int = Field(default=4, ge=1, le=16, description="Number of worker processes")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="json", pattern=r"^(json|plain)$")
    
    # Monitoring Configuration
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    enable_metrics: bool = Field(default=True)
    
    # Security Configuration
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_per_minute: int = Field(default=100, ge=1, le=1000)
    
    # Cache Configuration
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    cache_ttl: int = Field(default=300, ge=60, le=3600)
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()