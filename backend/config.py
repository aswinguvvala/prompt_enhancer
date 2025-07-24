import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the prompt enhancement system."""
    
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Model Configuration
    PROMPT_ENHANCER_MODEL: str = "microsoft/DialoGPT-medium"
    MAIN_LLM_MODEL: str = "gpt2"
    MAX_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    # Evaluation Configuration
    MIN_QUALITY_SCORE: float = 0.6
    ENABLE_QUALITY_CHECK: bool = True
    EVALUATION_METRICS: list = ["rouge", "bert_score", "length_ratio"]
    
    # Performance Configuration
    BATCH_SIZE: int = 4
    MAX_WORKERS: int = 4
    CACHE_SIZE: int = 1000
    REQUEST_TIMEOUT: int = 30
    
    # Security Configuration
    API_KEY: Optional[str] = None
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    
    # Environment Configuration
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
