"""
Model Management System for AI Prompt Enhancement
Handles model lifecycle, caching, and coordination
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from models.prompt_enhancer import PromptEnhancer, EnhancementResult
from config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    BUSY = "busy"

@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    backend: str
    status: ModelStatus
    load_time: Optional[float] = None
    last_used: Optional[float] = None
    usage_count: int = 0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class GenerationStats:
    """Statistics for text generation operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    total_tokens_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

class ModelManager:
    """Main model management system."""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.prompt_enhancer = PromptEnhancer()
        self.stats = GenerationStats()
        self._model_cache = {}
        self._loading_locks = {}
        
        # Initialize models based on configuration
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize model information based on configuration."""
        
        # Primary backend model
        primary_id = f"{settings.AI_BACKEND}_primary"
        self.models[primary_id] = ModelInfo(
            model_id=primary_id,
            backend=settings.AI_BACKEND,
            status=ModelStatus.UNLOADED
        )
        
        # Fallback backend model
        if settings.FALLBACK_BACKEND != settings.AI_BACKEND:
            fallback_id = f"{settings.FALLBACK_BACKEND}_fallback"
            self.models[fallback_id] = ModelInfo(
                model_id=fallback_id,
                backend=settings.FALLBACK_BACKEND,
                status=ModelStatus.UNLOADED
            )
    
    async def load_model(self, model_id: str) -> bool:
        """Load a specific model."""
        
        if model_id not in self.models:
            logger.error(f"Unknown model ID: {model_id}")
            return False
        
        model_info = self.models[model_id]
        
        # Check if already loaded
        if model_info.status == ModelStatus.READY:
            logger.info(f"Model {model_id} already loaded")
            return True
        
        # Prevent concurrent loading
        if model_id in self._loading_locks:
            logger.info(f"Model {model_id} is already being loaded, waiting...")
            await self._loading_locks[model_id].wait()
            return model_info.status == ModelStatus.READY
        
        # Create loading lock
        self._loading_locks[model_id] = asyncio.Event()
        
        try:
            logger.info(f"Loading model {model_id} ({model_info.backend})")
            model_info.status = ModelStatus.LOADING
            
            start_time = time.time()
            
            # Initialize the specific backend
            if model_info.backend == "transformers":
                await self.prompt_enhancer.transformers_client.initialize()
            elif model_info.backend == "ollama":
                # Ollama doesn't need explicit loading, but we can check availability
                if not await self.prompt_enhancer.ollama_client.is_available():
                    raise Exception("Ollama service not available")
            elif model_info.backend == "openai_compatible":
                # Check if the API is available
                if not await self.prompt_enhancer.openai_compatible_client.is_available():
                    raise Exception("OpenAI-compatible API not available")
            
            load_time = time.time() - start_time
            
            # Update model info
            model_info.status = ModelStatus.READY
            model_info.load_time = load_time
            model_info.error_message = None
            
            logger.info(f"Model {model_id} loaded successfully in {load_time:.2f}s")
            
            # Notify waiting coroutines
            self._loading_locks[model_id].set()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            model_info.status = ModelStatus.ERROR
            model_info.error_message = str(e)
            
            # Notify waiting coroutines
            self._loading_locks[model_id].set()
            return False
            
        finally:
            # Clean up loading lock
            if model_id in self._loading_locks:
                del self._loading_locks[model_id]
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a specific model."""
        
        if model_id not in self.models:
            logger.error(f"Unknown model ID: {model_id}")
            return False
        
        model_info = self.models[model_id]
        
        try:
            logger.info(f"Unloading model {model_id}")
            
            # Clear from cache
            if model_id in self._model_cache:
                del self._model_cache[model_id]
            
            # Update status
            model_info.status = ModelStatus.UNLOADED
            model_info.last_used = None
            
            logger.info(f"Model {model_id} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {str(e)}")
            model_info.error_message = str(e)
            return False
    
    async def enhance_prompt_with_management(self,
                                           context_injection_prompt: str,
                                           original_prompt: str,
                                           target_model: str,
                                           **kwargs) -> EnhancementResult:
        """Enhanced prompt processing with model management."""
        
        self.stats.total_requests += 1
        start_time = time.time()
        
        try:
            # Ensure primary model is loaded
            primary_id = f"{settings.AI_BACKEND}_primary"
            await self.load_model(primary_id)
            
            # Update model usage
            if primary_id in self.models:
                model_info = self.models[primary_id]
                model_info.status = ModelStatus.BUSY
                model_info.usage_count += 1
                model_info.last_used = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(context_injection_prompt, target_model, kwargs)
            
            if settings.ENABLE_MODEL_CACHING and cache_key in self._model_cache:
                cached_result = self._model_cache[cache_key]
                if time.time() - cached_result['timestamp'] < settings.MODEL_CACHE_TTL:
                    self.stats.cache_hits += 1
                    logger.info("Cache hit for prompt enhancement")
                    
                    # Update model status back to ready
                    if primary_id in self.models:
                        self.models[primary_id].status = ModelStatus.READY
                    
                    return EnhancementResult(**cached_result['result'])
            
            self.stats.cache_misses += 1
            
            # Perform enhancement
            result = await self.prompt_enhancer.enhance_prompt(
                context_injection_prompt=context_injection_prompt,
                original_prompt=original_prompt,
                target_model=target_model,
                **kwargs
            )
            
            # Cache the result
            if settings.ENABLE_MODEL_CACHING:
                self._model_cache[cache_key] = {
                    'result': asdict(result),
                    'timestamp': time.time()
                }
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats.successful_requests += 1
            self.stats.total_processing_time += processing_time
            self.stats.average_processing_time = (
                self.stats.total_processing_time / self.stats.successful_requests
            )
            
            # Estimate tokens (placeholder - would be replaced with real token counting)
            estimated_tokens = len(context_injection_prompt.split()) + len(result.enhanced_prompt.split())
            self.stats.total_tokens_processed += estimated_tokens
            
            # Update model status back to ready
            if primary_id in self.models:
                self.models[primary_id].status = ModelStatus.READY
            
            logger.info(f"Prompt enhancement completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {str(e)}")
            self.stats.failed_requests += 1
            
            # Update model status back to ready (or error if appropriate)
            primary_id = f"{settings.AI_BACKEND}_primary"
            if primary_id in self.models:
                self.models[primary_id].status = ModelStatus.ERROR if "model" in str(e).lower() else ModelStatus.READY
                self.models[primary_id].error_message = str(e)
            
            raise
    
    async def generate_alternatives_with_management(self,
                                                  context_injection_prompt: str,
                                                  original_prompt: str,
                                                  target_model: str,
                                                  num_alternatives: int = 3,
                                                  **kwargs) -> List[str]:
        """Generate alternatives with model management."""
        
        # Ensure model is loaded
        primary_id = f"{settings.AI_BACKEND}_primary"
        await self.load_model(primary_id)
        
        try:
            alternatives = await self.prompt_enhancer.generate_alternatives(
                context_injection_prompt=context_injection_prompt,
                original_prompt=original_prompt,
                target_model=target_model,
                num_alternatives=num_alternatives,
                **kwargs
            )
            
            return alternatives
            
        except Exception as e:
            logger.error(f"Alternative generation failed: {str(e)}")
            raise
    
    def _generate_cache_key(self, context_prompt: str, target_model: str, kwargs: Dict) -> str:
        """Generate a cache key for the request."""
        
        # Create a simplified hash of the important parameters
        import hashlib
        
        # Create a hash of the full context prompt to avoid cache key collisions
        prompt_hash = hashlib.md5(context_prompt.encode()).hexdigest()[:16]
        
        key_data = {
            'prompt_hash': prompt_hash,  # Use hash of full prompt instead of truncated
            'target_model': target_model,
            'temperature': kwargs.get('temperature', settings.DEFAULT_TEMPERATURE),
            'top_p': kwargs.get('top_p', settings.DEFAULT_TOP_P),
            'max_length': kwargs.get('max_length', settings.DEFAULT_MAX_LENGTH)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        
        current_time = time.time()
        expired_keys = []
        
        for key, cached_data in self._model_cache.items():
            if current_time - cached_data['timestamp'] > settings.MODEL_CACHE_TTL:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._model_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the model management system."""
        
        # Get backend health status
        backend_status = await self.prompt_enhancer.health_check()
        
        # Compile model status
        model_status = {}
        for model_id, model_info in self.models.items():
            model_status[model_id] = {
                'status': model_info.status.value,
                'backend': model_info.backend,
                'load_time': model_info.load_time,
                'last_used': model_info.last_used,
                'usage_count': model_info.usage_count,
                'error_message': model_info.error_message
            }
        
        # System statistics
        system_stats = {
            'total_requests': self.stats.total_requests,
            'successful_requests': self.stats.successful_requests,
            'failed_requests': self.stats.failed_requests,
            'success_rate': (
                self.stats.successful_requests / max(self.stats.total_requests, 1) * 100
            ),
            'average_processing_time': self.stats.average_processing_time,
            'total_tokens_processed': self.stats.total_tokens_processed,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'cache_hit_rate': (
                self.stats.cache_hits / max(self.stats.cache_hits + self.stats.cache_misses, 1) * 100
            ),
            'cache_size': len(self._model_cache)
        }
        
        return {
            'overall_status': 'healthy' if backend_status['primary_available'] else 'degraded',
            'backend_status': backend_status,
            'model_status': model_status,
            'system_stats': system_stats,
            'configuration': {
                'primary_backend': settings.AI_BACKEND,
                'fallback_backend': settings.FALLBACK_BACKEND,
                'caching_enabled': settings.ENABLE_MODEL_CACHING,
                'cache_ttl': settings.MODEL_CACHE_TTL,
                'max_alternatives': settings.MAX_ALTERNATIVES
            }
        }
    
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        
        if model_id not in self.models:
            return None
        
        model_info = self.models[model_id]
        return {
            'model_id': model_info.model_id,
            'backend': model_info.backend,
            'status': model_info.status.value,
            'load_time': model_info.load_time,
            'last_used': model_info.last_used,
            'usage_count': model_info.usage_count,
            'error_message': model_info.error_message,
            'metadata': model_info.metadata
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        return asdict(self.stats)
    
    async def shutdown(self):
        """Gracefully shutdown the model management system."""
        
        logger.info("Shutting down model management system...")
        
        # Unload all models
        for model_id in list(self.models.keys()):
            await self.unload_model(model_id)
        
        # Clear cache
        self._model_cache.clear()
        
        logger.info("Model management system shutdown complete")

# Global instance
model_manager = ModelManager()