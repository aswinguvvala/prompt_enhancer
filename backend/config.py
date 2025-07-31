import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the AI prompt enhancement system with real model integration."""
    
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8001  # Updated to match current running instance
    API_RELOAD: bool = True
    
    # AI Model Backend Configuration (OpenAI Only)
    AI_BACKEND: str = "openai"  # Only OpenAI backend supported
    FALLBACK_BACKEND: str = "openai"  # No fallback needed - OpenAI only
    
    # OpenAI Configuration (Primary and Only Backend)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o-mini"  # Cost-effective model for prompt enhancement
    OPENAI_TIMEOUT: int = 60  # Timeout for OpenAI API calls
    OPENAI_MAX_RETRIES: int = 3  # Number of retries for failed requests
    
    
    # Model Generation Parameters (Optimized for speed)
    DEFAULT_MAX_LENGTH: int = 400  # Further reduced for faster generation
    DEFAULT_TEMPERATURE: float = 0.6  # Slightly lower for more focused responses
    DEFAULT_TOP_P: float = 0.8  # Reduced for faster sampling
    DEFAULT_TOP_K: int = 30  # Further reduced for faster sampling
    CONTEXT_WINDOW_SIZE: int = 800  # Reduced for faster processing
    
    # Fast Model Configuration
    FAST_MODEL_MAX_LENGTH: int = 200  # For quick enhancements
    FAST_MODEL_TEMPERATURE: float = 0.5  # More deterministic for speed
    
    # Enhancement Configuration
    ENABLE_COMPREHENSIVE_GUIDES: bool = True
    MAX_CONTEXT_INJECTION_SIZE: int = 800  # Optimized for fast processing
    ENABLE_ALTERNATIVE_GENERATION: bool = True
    MAX_ALTERNATIVES: int = 5
    
    # Context Management Configuration
    ENABLE_SMART_CONTEXT_COMPRESSION: bool = True
    QUICK_ENHANCEMENT_RULE_LIMIT: int = 3  # Limit rules for quick enhancement
    CONTEXT_COMPRESSION_RATIO: float = 0.6  # Compress context to 60% of original size
    
    # Performance Configuration
    BATCH_SIZE: int = 4
    MAX_WORKERS: int = 4
    CACHE_SIZE: int = 1000
    REQUEST_TIMEOUT: int = 120  # Increased timeout for complex AI processing
    ENABLE_MODEL_CACHING: bool = True  # Re-enabled with fixed cache key implementation
    MODEL_CACHE_TTL: int = 3600  # Cache models for 1 hour
    
    # Advanced Timeout Configuration
    BASE_TIMEOUT: int = 30  # Base timeout for AI processing
    MAX_TIMEOUT: int = 120  # Maximum timeout for complex requests
    TIMEOUT_MULTIPLIER: float = 1.5  # Exponential backoff multiplier
    
    # Quality Assessment (AI-Based)
    ENABLE_AI_QUALITY_ASSESSMENT: bool = True
    QUALITY_ASSESSMENT_MODEL: str = "same"  # "same" uses enhancement model, or specify different
    MIN_QUALITY_THRESHOLD: float = 0.6
    
    # Security Configuration
    API_KEY: Optional[str] = None
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000", 
        "http://localhost:8080", 
        "http://localhost:8000",
        "http://localhost:8001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001",
        "null",  # Allow file:// origins (browsers send "null" for local files)
        "file://"  # Explicit file protocol support
    ]
    
    # Environment Configuration
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    ENABLE_DEBUG_LOGGING: bool = False
    
    # Advanced AI Model Settings
    ENABLE_STREAMING: bool = False  # For real-time response streaming
    ENABLE_FUNCTION_CALLING: bool = False  # Future feature for tool use
    MODEL_LOAD_TIMEOUT: int = 120  # Time to wait for model loading
    
    # Monitoring and Analytics
    ENABLE_METRICS: bool = True
    METRICS_ENDPOINT: Optional[str] = None
    TRACK_TOKEN_USAGE: bool = True
    TRACK_RESPONSE_TIMES: bool = True
    
    class Config:
        env_file = [".env", "../.env"]  # Check multiple locations
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
