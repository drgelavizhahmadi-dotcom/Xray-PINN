"""Configuration management."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    data_dir: Path = Path("./data")
    results_dir: Path = Path("./data/results")
    checkpoint_dir: Path = Path("./checkpoints")
    
    mc_dropout_samples: int = 30
    uncertainty_threshold_low: float = 0.10
    uncertainty_threshold_high: float = 0.25
    batch_size: int = 16
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    device_class: str = "IIa"
    intended_use: str = "chest_xray_screening"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
