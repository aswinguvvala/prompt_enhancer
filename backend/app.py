"""
AI Prompt Enhancement Studio - FastAPI Backend
Professional prompt optimization with real AI model integration
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import settings
from simplified_guides import SIMPLIFIED_MODEL_GUIDES
from models.prompt_enhancer import PromptEnhancer

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global prompt enhancer instance
prompt_enhancer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global prompt_enhancer
    
    # Startup
    logger.info("🚀 Starting AI Prompt Enhancement Studio...")
    
    try:
        # Initialize prompt enhancer
        prompt_enhancer = PromptEnhancer()
        logger.info("✅ AI model integration initialized successfully")
        
        # Perform health check
        health_status = await prompt_enhancer.health_check()
        if health_status.get("primary_available", False):
            logger.info("✅ Primary AI backend is healthy and ready")
        else:
            logger.warning("⚠️ Primary AI backend health check failed - using fallback mode")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI model integration: {str(e)}")
        # Continue startup with limited functionality
        
    logger.info("🎨 AI Prompt Enhancement Studio is ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Prompt Enhancement Studio...")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="AI Prompt Enhancement Studio",
    description="Professional-grade prompt optimization using real AI models with model-specific intelligence for OpenAI GPT-4, Anthropic Claude, Google Gemini, and xAI Grok",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware - Updated for better frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS + ["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"📨 {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    
    return response

# Pydantic models for API
class EnhanceRequest(BaseModel):
    original_prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt to enhance")
    target_model: str = Field(..., description="Target AI model (openai, claude, gemini, grok)")
    enhancement_type: str = Field(default="comprehensive", description="Enhancement level (quick, comprehensive)")
    max_length: Optional[int] = Field(default=None, description="Maximum output length")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="AI model temperature")

class EnhanceResponse(BaseModel):
    enhanced_prompt: str
    original_prompt: str
    target_model: str
    processing_time: float
    backend_used: str
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    backends: Dict[str, Any]
    models_available: list

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

# API Routes

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the main frontend application."""
    return FileResponse("frontend/index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with comprehensive system status."""
    try:
        health_data = await prompt_enhancer.health_check() if prompt_enhancer else {}
        
        return HealthResponse(
            status="healthy" if health_data.get("primary_available", False) else "degraded",
            version="2.0.0",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            backends=health_data.get("backends", {}),
            models_available=list(SIMPLIFIED_MODEL_GUIDES.keys())
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

@app.get("/status")
async def get_status():
    """Get detailed system status including performance metrics."""
    try:
        if not prompt_enhancer:
            raise HTTPException(status_code=503, detail="AI model integration not available")
            
        health_data = await prompt_enhancer.health_check()
        cache_stats = prompt_enhancer.get_cache_statistics()
        performance_stats = prompt_enhancer.get_performance_statistics()
        
        return {
            "system": {
                "status": "operational",
                "version": "2.0.0",
                "uptime": time.time(),
                "environment": settings.ENVIRONMENT
            },
            "ai_integration": {
                "backend": settings.AI_BACKEND,
                "model": settings.OPENAI_MODEL,
                "health": health_data
            },
            "performance": {
                "caching": cache_stats,
                "backend_stats": performance_stats
            },
            "models": {
                "available": list(SIMPLIFIED_MODEL_GUIDES.keys()),
                "default": "claude"
            }
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Unable to retrieve system status")

@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_prompt(request: EnhanceRequest):
    """Enhance a prompt using AI model with model-specific optimization rules."""
    start_time = time.time()
    
    logger.info(f"🎨 Enhancement request: {request.target_model}, length: {len(request.original_prompt)}")
    
    try:
        if not prompt_enhancer:
            raise HTTPException(status_code=503, detail="AI model integration not available")
        
        # Validate target model
        if request.target_model not in SIMPLIFIED_MODEL_GUIDES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported target model: {request.target_model}. Available: {list(SIMPLIFIED_MODEL_GUIDES.keys())}"
            )
        
        # Prepare enhancement parameters
        enhancement_params = {
            "temperature": request.temperature or settings.DEFAULT_TEMPERATURE,
            "max_length": request.max_length or settings.DEFAULT_MAX_LENGTH,
            "enhancement_type": request.enhancement_type
        }
        
        # Create context injection prompt using the system from streamlit_app.py
        context_prompt = create_context_injection_prompt(
            request.original_prompt,
            request.target_model,
            request.enhancement_type
        )
        
        # Enhance the prompt
        result = await prompt_enhancer.enhance_prompt(
            context_injection_prompt=context_prompt,
            original_prompt=request.original_prompt,
            target_model=request.target_model,
            **enhancement_params
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Enhancement completed in {processing_time:.2f}s using {result.backend_used}")
        
        return EnhanceResponse(
            enhanced_prompt=result.enhanced_prompt,
            original_prompt=result.original_prompt,
            target_model=request.target_model,
            processing_time=processing_time * 1000,  # Convert to milliseconds for frontend
            backend_used=result.backend_used,
            metadata={
                "context_injection_used": True,
                "enhancement_type": request.enhancement_type,
                "model_rules_applied": True,
                **result.metadata
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@app.post("/batch-enhance")
async def batch_enhance_prompts(requests: dict):
    """Enhance multiple prompts in batch."""
    prompts = requests.get("prompts", [])
    target_model = requests.get("target_model", "claude")
    
    if len(prompts) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 prompts per batch request")
    
    try:
        results = []
        for i, prompt_text in enumerate(prompts):
            if not prompt_text.strip():
                continue
                
            enhance_request = EnhanceRequest(
                original_prompt=prompt_text,
                target_model=target_model
            )
            
            result = await enhance_prompt(enhance_request)
            results.append({
                "index": i,
                "result": result
            })
        
        return {"batch_results": results, "total_processed": len(results)}
        
    except Exception as e:
        logger.error(f"Batch enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch enhancement failed: {str(e)}")

@app.get("/models/available")
async def get_available_models():
    """Get list of available AI models with their capabilities."""
    return {
        "models": SIMPLIFIED_MODEL_GUIDES,
        "default_model": "claude",
        "total_models": len(SIMPLIFIED_MODEL_GUIDES)
    }

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    if model_name not in SIMPLIFIED_MODEL_GUIDES:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "model": model_name,
        "info": SIMPLIFIED_MODEL_GUIDES[model_name],
        "optimization_rules": len(SIMPLIFIED_MODEL_GUIDES[model_name].get("rules", [])),
        "supported": True
    }

# Utility functions (from streamlit_app.py)
def create_context_injection_prompt(original_prompt: str, target_model: str, enhancement_type: str) -> str:
    """Create comprehensive context injection prompt using research-based meta-prompt template system."""
    
    logger.info(f"Creating context injection for target_model: {target_model}, enhancement_type: {enhancement_type}")
    
    if target_model not in SIMPLIFIED_MODEL_GUIDES:
        logger.warning(f"Target model {target_model} not found in SIMPLIFIED_MODEL_GUIDES")
        return original_prompt
        
    guide = SIMPLIFIED_MODEL_GUIDES[target_model]
    model_name = guide['name']
    model_description = guide['description']
    model_rules = guide.get('rules', [])
    output_guidelines = guide.get('output_guidelines', [])
    avoid_rules = guide.get('avoid', [])
    
    # Format rule sections with clear structure
    rules_section = "\\n".join([f"{i+1}. {rule}" for i, rule in enumerate(model_rules)])
    output_section = "\\n".join([f"• {guideline}" for guideline in output_guidelines])
    avoid_section = "\\n".join([f"• {rule}" for rule in avoid_rules])
    
    # Anti-XML instructions for natural language output
    from simplified_guides import ANTI_XML_INSTRUCTIONS, ENHANCEMENT_INSTRUCTIONS
    enhancement_instructions = ENHANCEMENT_INSTRUCTIONS[:5] if enhancement_type == "comprehensive" else ENHANCEMENT_INSTRUCTIONS[:3]
    enhancement_section = "\\n".join([f"• {instruction}" for instruction in enhancement_instructions])
    anti_xml_section = "\\n".join([f"• {instruction}" for instruction in ANTI_XML_INSTRUCTIONS[:4]])
    
    # Create comprehensive meta-prompt template
    if enhancement_type == "quick":
        context_prompt = f"""[INST]
You are an expert AI Prompt Engineer specializing in {model_name} optimization. Your task is to quickly enhance a user's prompt using proven best practices.

TARGET MODEL: {model_name}
DESCRIPTION: {model_description}

KEY OPTIMIZATION RULES:
{rules_section}

WHAT TO AVOID:
{avoid_section}

ENHANCEMENT GUIDELINES:
{enhancement_section}

NO TECHNICAL MARKUP:
{anti_xml_section}

USER'S ORIGINAL PROMPT:
"{original_prompt}"

Create a significantly improved version that applies the optimization rules above. Write in natural language only - no XML tags, technical markup, or structured formatting.

ENHANCED PROMPT:
[/INST]"""
    else:
        context_prompt = f"""[INST]
You are an expert AI Prompt Engineer specializing in {model_name} optimization. Transform the user's raw prompt into a highly effective version using comprehensive, research-based best practices.

TARGET MODEL: {model_name}
DESCRIPTION: {model_description}

COMPREHENSIVE OPTIMIZATION RULES FOR {model_name.upper()}:
{rules_section}

OUTPUT GUIDELINES:
{output_section}

CRITICAL - WHAT TO AVOID:
{avoid_section}

ENHANCEMENT PROCESS:
{enhancement_section}

NO TECHNICAL MARKUP REQUIREMENTS:
{anti_xml_section}

USER'S ORIGINAL PROMPT TO ENHANCE:
"{original_prompt}"

Your task is to create a dramatically improved version that follows all the {model_name} optimization principles above. Apply systematic enhancement techniques including role assignment, context enrichment, example provision, structured formatting requests, and success criteria specification where appropriate.

Write your enhanced prompt using clear, natural language without any XML tags, angle brackets, or programming-style syntax. The result should be a highly optimized prompt that will generate significantly better responses from {model_name}.

ENHANCED PROMPT:
[/INST]"""
    
    return context_prompt

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=exc.detail,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An unexpected error occurred",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        ).dict()
    )

# Serve static files (CSS, JS, images)
app.mount("/frontend", StaticFiles(directory="frontend"), name="static")
app.mount("/static", StaticFiles(directory="frontend"), name="static_alias")

# Development server
if __name__ == "__main__":
    logger.info("🚀 Starting AI Prompt Enhancement Studio in development mode...")
    
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )