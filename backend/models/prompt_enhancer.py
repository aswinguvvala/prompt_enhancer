"""
Real AI Model Integration for Prompt Enhancement
OpenAI-only integration for streamlined deployment
"""

import asyncio
import aiohttp
import json
import logging
import hashlib
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

class PerformanceCache:
    """Smart caching system for AI model responses and contexts."""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self._response_cache = {}  # prompt_hash -> (response, timestamp)
        self._context_cache = {}   # context_hash -> (context, timestamp)
        self._hit_stats = {"hits": 0, "misses": 0}
        
    def _create_hash(self, text: str, params: Dict[str, Any] = None) -> str:
        """Create a hash for caching keys."""
        content = text
        if params:
            # Include relevant parameters in hash
            param_str = json.dumps(sorted(params.items()), default=str)
            content += f"|{param_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.ttl
    
    def _cleanup_expired(self):
        """Remove expired entries from both caches."""
        current_time = time.time()
        
        # Clean response cache
        expired_responses = [
            key for key, (_, timestamp) in self._response_cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_responses:
            del self._response_cache[key]
            
        # Clean context cache
        expired_contexts = [
            key for key, (_, timestamp) in self._context_cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_contexts:
            del self._context_cache[key]
            
        # If still too large, remove oldest entries
        if len(self._response_cache) > self.max_size:
            sorted_items = sorted(
                self._response_cache.items(), 
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            to_remove = len(sorted_items) - self.max_size
            for key, _ in sorted_items[:to_remove]:
                del self._response_cache[key]
    
    def get_response(self, prompt: str, target_model: str, params: Dict[str, Any] = None) -> Optional[str]:
        """Get cached response if available."""
        cache_key = self._create_hash(f"{prompt}|{target_model}", params)
        
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if self._is_valid(timestamp):
                self._hit_stats["hits"] += 1
                logger.info(f"Cache HIT for prompt hash: {cache_key[:8]}...")
                return response
            else:
                del self._response_cache[cache_key]
        
        self._hit_stats["misses"] += 1
        logger.info(f"Cache MISS for prompt hash: {cache_key[:8]}...")
        return None
    
    def store_response(self, prompt: str, target_model: str, response: str, params: Dict[str, Any] = None):
        """Store response in cache."""
        self._cleanup_expired()
        
        cache_key = self._create_hash(f"{prompt}|{target_model}", params)
        self._response_cache[cache_key] = (response, time.time())
        logger.info(f"Cached response for hash: {cache_key[:8]}...")
    
    def get_context(self, context_signature: str) -> Optional[str]:
        """Get cached context if available."""
        cache_key = self._create_hash(context_signature)
        
        if cache_key in self._context_cache:
            context, timestamp = self._context_cache[cache_key]
            if self._is_valid(timestamp):
                logger.info(f"Context cache HIT for: {cache_key[:8]}...")
                return context
            else:
                del self._context_cache[cache_key]
        
        return None
    
    def store_context(self, context_signature: str, context: str):
        """Store prepared context in cache."""
        self._cleanup_expired()
        
        cache_key = self._create_hash(context_signature)
        self._context_cache[cache_key] = (context, time.time())
        logger.info(f"Cached context for: {cache_key[:8]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._hit_stats["hits"] + self._hit_stats["misses"]
        hit_rate = self._hit_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "total_hits": self._hit_stats["hits"],
            "total_misses": self._hit_stats["misses"],
            "response_cache_size": len(self._response_cache),
            "context_cache_size": len(self._context_cache),
            "cache_ttl": self.ttl
        }

class ModelPerformanceTracker:
    """Track AI model performance and automatically optimize."""
    
    def __init__(self):
        self.stats = {}  # backend_name -> {"response_times": [], "errors": 0, "successes": 0}
        
    def record_response_time(self, backend: str, response_time: float):
        """Record successful response time."""
        if backend not in self.stats:
            self.stats[backend] = {"response_times": [], "errors": 0, "successes": 0}
        
        self.stats[backend]["response_times"].append(response_time)
        self.stats[backend]["successes"] += 1
        
        # Keep only last 100 response times to prevent memory bloat
        if len(self.stats[backend]["response_times"]) > 100:
            self.stats[backend]["response_times"] = self.stats[backend]["response_times"][-100:]
    
    def record_error(self, backend: str):
        """Record error for backend."""
        if backend not in self.stats:
            self.stats[backend] = {"response_times": [], "errors": 0, "successes": 0}
        
        self.stats[backend]["errors"] += 1
    
    def get_best_backend(self, available_backends: List[str]) -> str:
        """Get the best performing backend from available options."""
        best_backend = available_backends[0]  # Default fallback
        best_score = 0
        
        for backend in available_backends:
            if backend not in self.stats:
                continue
                
            stats = self.stats[backend]
            total_requests = stats["successes"] + stats["errors"]
            
            if total_requests == 0:
                continue
                
            # Calculate performance score (success rate + speed bonus)
            success_rate = stats["successes"] / total_requests
            avg_response_time = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 10.0
            
            # Score: success rate * speed bonus (faster = higher score)
            speed_bonus = max(0.1, 1.0 / (avg_response_time / 5.0))  # 5s is baseline
            score = success_rate * speed_bonus
            
            if score > best_score:
                best_score = score
                best_backend = backend
        
        logger.info(f"Best backend selected: {best_backend} (score: {best_score:.3f})")
        return best_backend
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all backends."""
        summary = {}
        for backend, stats in self.stats.items():
            total_requests = stats["successes"] + stats["errors"]
            success_rate = stats["successes"] / total_requests if total_requests > 0 else 0
            avg_response_time = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
            
            summary[backend] = {
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "total_requests": total_requests,
                "errors": stats["errors"]
            }
        
        return summary

@dataclass
class EnhancementResult:
    """Result of prompt enhancement operation."""
    enhanced_prompt: str
    original_prompt: str
    model_used: str
    backend_used: str
    processing_time: float
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None



class OpenAIClient:
    """Client for OpenAI API using official Python library."""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.timeout = settings.OPENAI_TIMEOUT
        self.max_retries = settings.OPENAI_MAX_RETRIES
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        
        logger.info(f"=== OPENAI API CALL DEBUG ===")
        logger.info(f"Model: {self.model}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Prompt content preview: {prompt[:500]}...")
        logger.info(f"Parameters: max_tokens={kwargs.get('max_length', settings.DEFAULT_MAX_LENGTH)}, temperature={kwargs.get('temperature', settings.DEFAULT_TEMPERATURE)}, top_p={kwargs.get('top_p', settings.DEFAULT_TOP_P)}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": kwargs.get("max_length", settings.DEFAULT_MAX_LENGTH),
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", settings.DEFAULT_TOP_P),
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_content = result["choices"][0]["message"]["content"].strip()
                            logger.info(f"=== OPENAI API RESPONSE DEBUG ===")
                            logger.info(f"Response length: {len(response_content)} characters")
                            logger.info(f"Response content preview: {response_content[:500]}...")
                            logger.info(f"Full response (truncated): {response_content[:1000]}...")
                            logger.info(f"=== END OPENAI DEBUG ===")
                            return response_content
                        else:
                            error_text = await response.text()
                            logger.error(f"OpenAI API error (attempt {attempt + 1}): {response.status} - {error_text}")
                            
            except asyncio.TimeoutError:
                logger.error(f"OpenAI timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"OpenAI error (attempt {attempt + 1}): {str(e)}")
                
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        raise Exception(f"OpenAI API failed after {self.max_retries} attempts")
    
    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key:
            return False
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get("https://api.openai.com/v1/models", headers=headers) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"OpenAI availability check failed: {str(e)}")
            return False


class PromptEnhancer:
    """Main prompt enhancement engine with OpenAI integration and intelligent caching."""
    
    def __init__(self):
        # OpenAI-only configuration
        self.primary_backend = "openai"
        self.fallback_backend = "openai"
        
        # Initialize OpenAI client only
        self.openai_client = OpenAIClient()
        
        # Initialize performance systems
        self.cache = PerformanceCache(
            ttl=settings.MODEL_CACHE_TTL if settings.ENABLE_MODEL_CACHING else 300,  # 5 min minimum
            max_size=settings.CACHE_SIZE
        )
        self.performance_tracker = ModelPerformanceTracker()
        
        # Legacy model cache (keeping for compatibility)
        self._model_cache = {}
        
        logger.info("PromptEnhancer initialized with OpenAI-only integration and intelligent caching")
        
    async def enhance_prompt(self, 
                           context_injection_prompt: str,
                           original_prompt: str,
                           target_model: str,
                           **kwargs) -> EnhancementResult:
        """Enhance a prompt using real AI model with intelligent caching and performance optimization."""
        
        import time
        start_time = time.time()
        
        # Create cache parameters (exclude non-deterministic fields)
        cache_params = {
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", settings.DEFAULT_TOP_P),
            "max_length": kwargs.get("max_length", settings.DEFAULT_MAX_LENGTH),
            "enhancement_type": kwargs.get("enhancement_type", "general")
        }
        
        # Check cache first for exact prompt + target model combination
        if settings.ENABLE_MODEL_CACHING:
            cached_response = self.cache.get_response(
                original_prompt, 
                target_model, 
                cache_params
            )
            
            if cached_response:
                # Extract enhanced prompt from cached response
                enhanced_prompt = self._extract_enhanced_prompt(cached_response, original_prompt)
                
                logger.info(f"Using cached enhancement for {target_model}")
                return EnhancementResult(
                    enhanced_prompt=enhanced_prompt,
                    original_prompt=original_prompt,
                    model_used=target_model,
                    backend_used="cache",
                    processing_time=time.time() - start_time,
                    metadata={
                        "cache_hit": True,
                        "context_injection_used": True,
                        "ai_response_length": len(cached_response),
                        "extraction_successful": enhanced_prompt != original_prompt
                    }
                )
        
        # DEBUG: Log what we're sending to the AI model
        logger.info(f"=== CONTEXT INJECTION DEBUG ===")
        logger.info(f"Target Model: {target_model}")
        logger.info(f"Original Prompt: {original_prompt}")
        logger.info(f"Context Injection Prompt Length: {len(context_injection_prompt)}")
        logger.info(f"Context Injection Preview: {context_injection_prompt[:500]}...")
        
        # Determine best backend based on performance history
        available_backends = [self.primary_backend, self.fallback_backend]
        if len(available_backends) > 1:
            best_backend = self.performance_tracker.get_best_backend(available_backends)
        else:
            best_backend = self.primary_backend
        
        backend_used = None
        enhanced_text = None
        
        # Try best backend first
        try:
            backend_attempt_start = time.time()
            backend_used, enhanced_text = await self._try_backend(
                best_backend, 
                context_injection_prompt, 
                **kwargs
            )
            
            # Record successful performance
            backend_time = time.time() - backend_attempt_start
            self.performance_tracker.record_response_time(best_backend, backend_time)
            
            # DEBUG: Log what we got back
            logger.info(f"AI Response Length: {len(enhanced_text)}")
            logger.info(f"AI Response Preview: {enhanced_text[:300]}...")
            
        except Exception as e:
            # Record error for primary backend
            self.performance_tracker.record_error(best_backend)
            logger.warning(f"Best backend ({best_backend}) failed: {str(e)}")
            
            # Try other available backends
            remaining_backends = [b for b in available_backends if b != best_backend]
            
            for fallback_backend in remaining_backends:
                try:
                    backend_attempt_start = time.time()
                    backend_used, enhanced_text = await self._try_backend(
                        fallback_backend, 
                        context_injection_prompt, 
                        **kwargs
                    )
                    
                    # Record successful performance for fallback
                    backend_time = time.time() - backend_attempt_start
                    self.performance_tracker.record_response_time(fallback_backend, backend_time)
                    
                    logger.info(f"Fallback AI Response Length: {len(enhanced_text)}")
                    logger.info(f"Fallback AI Response Preview: {enhanced_text[:300]}...")
                    break
                    
                except Exception as fallback_error:
                    self.performance_tracker.record_error(fallback_backend)
                    logger.error(f"Fallback backend ({fallback_backend}) also failed: {str(fallback_error)}")
                    continue
            
            # If all backends failed
            if enhanced_text is None:
                logger.error("All AI backends failed, using rule-based fallback enhancement")
                enhanced_text = self._apply_rule_based_enhancement(original_prompt, target_model)
                backend_used = "rule_based_fallback"
        
        processing_time = time.time() - start_time
        
        # Extract the actual enhanced prompt from the AI response
        enhanced_prompt = self._extract_enhanced_prompt(enhanced_text, original_prompt)
        
        # Cache the successful response (only if from real AI model)
        if settings.ENABLE_MODEL_CACHING and backend_used not in ["rule_based_fallback", "manual_fallback"]:
            self.cache.store_response(
                original_prompt, 
                target_model, 
                enhanced_text, 
                cache_params
            )
        
        # DEBUG: Log extraction results
        logger.info(f"Extracted Enhanced Prompt: {enhanced_prompt}")
        logger.info(f"Enhancement Different from Original: {enhanced_prompt != original_prompt}")
        logger.info(f"=== END DEBUG ===")
        
        return EnhancementResult(
            enhanced_prompt=enhanced_prompt,
            original_prompt=original_prompt,
            model_used=target_model,
            backend_used=backend_used,
            processing_time=processing_time,
            metadata={
                "context_injection_used": True,
                "cache_hit": False,
                "backend_attempted": available_backends,
                "ai_response_length": len(enhanced_text),
                "extraction_successful": enhanced_prompt != original_prompt
            }
        )
    
    async def _try_backend(self, backend: str, prompt: str, **kwargs) -> tuple[str, str]:
        """Try OpenAI backend for text generation."""
        
        if backend == "openai":
            if await self.openai_client.is_available():
                enhanced_text = await self.openai_client.generate(prompt, **kwargs)
                return "openai", enhanced_text
            else:
                raise Exception("OpenAI API not available")
        else:
            raise Exception(f"Only OpenAI backend is supported, got: {backend}")
    
    def _apply_rule_based_enhancement(self, original_prompt: str, target_model: str) -> str:
        """Apply rule-based prompt enhancement when AI models are unavailable."""
        
        logger.info(f"Applying rule-based enhancement for {target_model}")
        
        # Import re for text processing
        import re
        
        # Clean and analyze the original prompt
        prompt = original_prompt.strip()
        
        # Basic enhancements based on target model
        enhanced_parts = []
        
        # Add role/context if missing
        if not any(role in prompt.lower() for role in ['you are', 'act as', 'as a', 'you\'re']):
            if target_model == "openai":
                enhanced_parts.append("You are an expert assistant.")
            elif target_model == "claude":
                enhanced_parts.append("I need you to help me as a knowledgeable assistant.")
            elif target_model == "gemini":
                enhanced_parts.append("As an AI assistant,")
            elif target_model == "grok":
                enhanced_parts.append("Hey, I need your help with this.")
        
        # Add clarity and structure
        if '?' not in prompt and not prompt.lower().startswith(('please', 'can you', 'could you', 'would you')):
            if target_model == "openai":
                enhanced_parts.append("Please")
            elif target_model == "claude":
                enhanced_parts.append("Could you please")
            elif target_model == "gemini":
                enhanced_parts.append("I would like you to")
            elif target_model == "grok":
                enhanced_parts.append("Can you")
        
        # Enhance the main prompt content
        enhanced_prompt = prompt
        
        # Add specificity requests
        specificity_requests = []
        if len(prompt.split()) < 10:  # Short prompts need more detail requests
            if target_model == "openai":
                specificity_requests.append("Please provide detailed explanations and examples.")
            elif target_model == "claude":
                specificity_requests.append("I'd appreciate comprehensive details in your response.")
            elif target_model == "gemini":
                specificity_requests.append("Please include specific details and context.")
            elif target_model == "grok":
                specificity_requests.append("Give me the full breakdown with details.")
        
        # Add format guidance if none exists
        if not any(fmt in prompt.lower() for fmt in ['format', 'structure', 'organize', 'list', 'steps']):
            if target_model == "openai":
                specificity_requests.append("Structure your response clearly.")
            elif target_model == "claude":
                specificity_requests.append("Please organize your response in a clear, helpful way.")
            elif target_model == "gemini":
                specificity_requests.append("Format your response for clarity.")
            elif target_model == "grok":
                specificity_requests.append("Make it clear and easy to follow.")
        
        # Combine all parts
        result_parts = []
        
        if enhanced_parts:
            result_parts.extend(enhanced_parts)
        
        result_parts.append(enhanced_prompt)
        
        if specificity_requests:
            result_parts.extend(specificity_requests)
        
        # Join with appropriate spacing
        enhanced_result = " ".join(result_parts)
        
        # Ensure proper punctuation
        if not enhanced_result.endswith(('.', '?', '!')):
            enhanced_result += "."
        
        logger.info(f"Rule-based enhancement: {len(original_prompt)} → {len(enhanced_result)} chars")
        logger.info(f"Enhanced prompt preview: {enhanced_result[:200]}...")
        
        return enhanced_result
    
    def _extract_enhanced_prompt(self, ai_response: str, original_prompt: str) -> str:
        """Extract the enhanced prompt from AI response with comprehensive pattern matching for meta-prompt formats."""
        
        logger.info(f"=== EXTRACTION DEBUG ===")
        logger.info(f"Original prompt length: {len(original_prompt)}")
        logger.info(f"AI response length: {len(ai_response)}")
        logger.info(f"AI response preview: {ai_response[:300]}...")
        
        # Clean up the response first
        ai_response = ai_response.strip()
        
        # Strategy 1: Look for meta-prompt format responses (for our comprehensive system)
        import re
        
        # Handle [INST]...[/INST] format responses - the response is everything after [/INST]
        if "[/INST]" in ai_response:
            after_inst = ai_response.split("[/INST]", 1)[-1].strip()
            if after_inst and len(after_inst) > len(original_prompt) * 0.5:
                # Clean up any remaining artifacts
                after_inst = re.sub(r'^\s*ENHANCED PROMPT:\s*', '', after_inst, flags=re.IGNORECASE)
                after_inst = re.sub(r'^\s*Enhanced:\s*', '', after_inst, flags=re.IGNORECASE)
                after_inst = after_inst.strip()
                if after_inst and after_inst != original_prompt:
                    logger.info(f"Found enhanced prompt after [/INST]: {after_inst[:100]}...")
                    return after_inst
        
        # Strategy 2: Look for "ENHANCED PROMPT:" marker (our meta-prompt template)
        enhanced_prompt_pattern = r'ENHANCED PROMPT:\s*(.+?)(?:\n\n|\Z)'
        match = re.search(enhanced_prompt_pattern, ai_response, re.IGNORECASE | re.DOTALL)
        if match:
            enhanced_text = match.group(1).strip()
            if enhanced_text and enhanced_text != original_prompt and len(enhanced_text) > 10:
                logger.info(f"Found enhanced prompt after marker: {enhanced_text[:100]}...")
                return enhanced_text
        
        # Strategy 2.1: Look for content after "ENHANCED PROMPT:" (everything after the marker)
        if "ENHANCED PROMPT:" in ai_response.upper():
            parts = ai_response.upper().split("ENHANCED PROMPT:")
            if len(parts) > 1:
                # Get the original case version
                original_parts = ai_response.split("ENHANCED PROMPT:", 1)
                if len(original_parts) > 1:
                    enhanced_text = original_parts[1].strip()
                    if enhanced_text and enhanced_text != original_prompt and len(enhanced_text) > 10:
                        logger.info(f"Found enhanced prompt after ENHANCED PROMPT marker: {enhanced_text[:100]}...")
                        return enhanced_text
        
        # Strategy 3: Look for quoted content (most reliable for prompt responses)
        quoted_patterns = [
            r'"([^"]{30,})"',  # Standard double quotes
            r"'([^']{30,})'",  # Single quotes
            r'`([^`]{30,})`',  # Backticks
        ]
        
        for pattern in quoted_patterns:
            quoted_matches = re.findall(pattern, ai_response)
            for match in quoted_matches:
                match = match.strip()
                # Skip if it's just the original prompt or too similar
                if (match != original_prompt and 
                    len(match) > len(original_prompt) * 0.7 and
                    not match.lower().startswith(('here is', 'here\'s', 'this is', 'changes made'))):
                    logger.info(f"Found enhanced prompt in quotes: {match[:100]}...")
                    return match
        
        # Strategy 2: Look for content after specific markers
        lines = ai_response.split('\n')
        enhanced_sections = []
        capture = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Start capturing after enhanced prompt markers
            if any(marker in line.lower() for marker in [
                "enhanced prompt for", "enhanced prompt:", "enhanced:", 
                "optimized prompt:", "improved prompt:", "rewritten prompt:",
                "better version:", "optimized version:"
            ]):
                capture = True
                
                # If the line contains the prompt content after the marker, extract it
                for marker in ["enhanced prompt for", "enhanced prompt:", "enhanced:", 
                              "optimized prompt:", "improved prompt:", "rewritten prompt:"]:
                    if marker in line.lower():
                        after_marker = line[line.lower().find(marker) + len(marker):].strip()
                        # Remove any remaining markers like "GPT-4:" 
                        after_marker = re.sub(r'^[A-Z-]+:\s*', '', after_marker)
                        if after_marker and len(after_marker) > 20:
                            enhanced_sections.append(after_marker)
                        break
                continue
            
            # Stop capturing at explanation markers
            if capture and any(marker in line.lower() for marker in [
                "changes made:", "explanation:", "reasoning:", "analysis:", 
                "note:", "why this works:", "breakdown:", "summary:", 
                "tips:", "additional:", "modifications:", "here's what"
            ]):
                break
                
            # Capture meaningful lines while in capture mode
            if capture and line and not line.startswith(('*', '-', '•', '1.', '2.', '3.')):
                # Clean up any remaining formatting
                clean_line = re.sub(r'^[""\'`]+|[""\'`]+$', '', line).strip()
                if clean_line:
                    enhanced_sections.append(clean_line)
        
        # Process captured sections
        if enhanced_sections:
            enhanced_prompt = ' '.join(enhanced_sections).strip()
            
            # Clean up common AI response artifacts
            enhanced_prompt = re.sub(r'^[""\'`]+|[""\'`]+$', '', enhanced_prompt)
            enhanced_prompt = enhanced_prompt.replace('"""', '').replace('```', '').strip()
            
            # Remove common prefixes
            prefixes_to_remove = [
                "here is the enhanced prompt:",
                "here's the enhanced prompt:",
                "here is the",
                "here's the",
                "the enhanced version is:",
                "enhanced version:"
            ]
            
            for prefix in prefixes_to_remove:
                if enhanced_prompt.lower().startswith(prefix):
                    enhanced_prompt = enhanced_prompt[len(prefix):].strip()
            
            if enhanced_prompt and enhanced_prompt != original_prompt and len(enhanced_prompt) > 20:
                logger.info(f"Successfully extracted enhanced prompt: {enhanced_prompt[:100]}...")
                return enhanced_prompt
        
        # Strategy 3: Look for the first substantial paragraph that's not explanation
        paragraphs = ai_response.split('\n\n')
        for section in paragraphs:
            section = section.strip()
            
            # Skip explanation paragraphs
            if any(section.lower().startswith(skip) for skip in [
                'to enhance', 'by providing', 'changes made', 'modifications',
                'these modifications', 'the inclusion', 'i\'ll consider',
                'gpt-4 excels', 'however', 'explanation', 'analysis'
            ]):
                continue
                
            # Look for prompt-like content
            if (len(section) > len(original_prompt) * 0.5 and 
                section != original_prompt and
                section.count(' ') > 5):  # Ensure it's substantial
                
                # Clean up the section
                section = re.sub(r'^[""\'`]+|[""\'`]+$', '', section).strip()
                
                if section and section != original_prompt:
                    logger.info(f"Found enhanced prompt in paragraph: {section[:100]}...")
                    return section
        
        # Strategy 4: Extract first sentence/line that looks like an enhanced prompt
        sentences = re.split(r'[.!?]+\s+', ai_response)
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip if it's too short or clearly explanatory
            if (len(sentence) > len(original_prompt) * 0.7 and
                not sentence.lower().startswith(('to enhance', 'by providing', 'gpt-4', 
                                                'changes made', 'however', 'the inclusion')) and
                sentence != original_prompt):
                
                sentence = re.sub(r'^[""\'`]+|[""\'`]+$', '', sentence).strip()
                if sentence:
                    logger.info(f"Found enhanced prompt in sentence: {sentence[:100]}...")
                    return sentence

        # Final fallback: return the AI response as-is if extraction fails
        logger.warning(f"Could not extract enhanced prompt, returning AI response as-is")
        # If the AI response is reasonable length and different from original, use it
        if (ai_response != original_prompt and 
            len(ai_response.strip()) > 10 and 
            len(ai_response) < len(original_prompt) * 5):  # Sanity check on length
            logger.info(f"=== EXTRACTION RESULT: AI RESPONSE AS-IS ===")
            logger.info(f"Returning AI response: {ai_response[:100]}...")
            return ai_response.strip()
        else:
            # Truly final fallback - return original
            logger.info(f"=== EXTRACTION RESULT: ORIGINAL UNCHANGED ===")
            return original_prompt
    
    async def generate_alternatives(self, 
                                  context_injection_prompt: str,
                                  original_prompt: str,
                                  target_model: str,
                                  num_alternatives: int = 3,
                                  **kwargs) -> List[str]:
        """Generate multiple alternative enhanced prompts."""
        
        alternatives = []
        
        for i in range(num_alternatives):
            # Modify the context injection prompt for variety
            varied_prompt = context_injection_prompt.replace(
                "ENHANCED PROMPT:",
                f"ENHANCED PROMPT (Alternative {i+1} with different approach):"
            )
            
            # Add variety instruction
            if i > 0:
                varied_prompt += f"\n\nNote: This is alternative {i+1}. Use a different optimization approach than previous versions."
            
            try:
                # Create modified kwargs with increased temperature for variety
                alt_kwargs = kwargs.copy()
                base_temp = alt_kwargs.get("temperature", 0.8)
                alt_kwargs["temperature"] = min(1.0, base_temp + (i * 0.1))  # Increase temperature for variety, cap at 1.0
                
                result = await self.enhance_prompt(
                    varied_prompt, original_prompt, target_model, 
                    **alt_kwargs
                )
                alternatives.append(result.enhanced_prompt)
                
            except Exception as e:
                logger.error(f"Failed to generate alternative {i+1}: {str(e)}")
                # Add a basic alternative
                alternatives.append(f"Alternative enhanced version {i+1}: {original_prompt}")
        
        return alternatives
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of OpenAI backend with performance statistics."""
        
        # OpenAI backend availability only
        backend_status = {
            "openai": {
                "available": await self.openai_client.is_available(),
                "model": self.openai_client.model,
                "api_configured": bool(self.openai_client.api_key)
            }
        }
        
        # Performance and caching statistics
        cache_stats = self.cache.get_stats()
        performance_stats = self.performance_tracker.get_performance_stats()
        
        # Compile comprehensive status
        status = {
            "backends": backend_status,
            "primary_backend": self.primary_backend,
            "fallback_backend": self.fallback_backend,
            "primary_available": backend_status["openai"]["available"],
            "fallback_available": backend_status["openai"]["available"],
            "caching": {
                "enabled": settings.ENABLE_MODEL_CACHING,
                "statistics": cache_stats
            },
            "performance": {
                "tracking_enabled": True,
                "backend_stats": performance_stats
            }
        }
        
        return status
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache performance statistics."""
        return self.cache.get_stats()
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get detailed backend performance statistics."""
        return self.performance_tracker.get_performance_stats()
    
    def clear_cache(self):
        """Clear all cached responses and contexts."""
        self.cache._response_cache.clear()
        self.cache._context_cache.clear()
        self.cache._hit_stats = {"hits": 0, "misses": 0}
        logger.info("All caches cleared manually")
    
    def optimize_cache(self):
        """Manually trigger cache optimization and cleanup."""
        self.cache._cleanup_expired()
        logger.info("Cache optimization completed")
    
    async def _call_ai_model(self, prompt: str, **kwargs) -> str:
        """Internal method to call OpenAI API directly."""
        try:
            backend_used, response = await self._try_backend(
                "openai", 
                prompt, 
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise Exception("OpenAI API unavailable")

# Global instance
prompt_enhancer = PromptEnhancer()