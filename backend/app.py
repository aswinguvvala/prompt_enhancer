"""
FastAPI Backend for AI Prompt Enhancement Studio with Multi-Model Support
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from config import settings

# ===== MODEL-SPECIFIC OPTIMIZATION RULES =====
MODEL_RULES = {
    "openai": {
        "name": "OpenAI GPT-4",
        "rules": [
            "Be explicit and specific with clear instructions",
            "Use iterative refinement approach for complex tasks",
            "Structure prompts with delimiters and format preferences", 
            "Implement chain-of-thought reasoning with 'Think step by step'",
            "Consider recency bias - place important info at the end",
            "Include specific examples to show desired patterns",
            "Specify output format explicitly (JSON, markdown, etc.)",
            "Use role definition at the beginning of prompts"
        ],
        "enhancements": {
            "prefix": "You are an expert assistant. Please approach this systematically:\n\n",
            "suffix": "\n\nPlease think step by step and provide a detailed, well-structured response.",
            "structure": "role-context-task-format",
            "examples": True,
            "chain_of_thought": True
        }
    },
    "claude": {
        "name": "Anthropic Claude",
        "rules": [
            "Use XML tags for structure (<example>, <document>, <thinking>)",
            "Implement multishot prompting with 3-5 diverse examples",
            "Include chain-of-thought reasoning before final answers",
            "Use prefilling to guide output format",
            "Provide context and motivation behind instructions",
            "Break complex tasks into smaller subtasks",
            "Structure: Task context → Tone → Background → Detailed task",
            "Use explicit instructions with clear explanations"
        ],
        "enhancements": {
            "prefix": "<task>\nYou are an expert assistant who thinks carefully before responding.\n",
            "suffix": "\n</task>\n\nPlease think through this step by step in <thinking> tags, then provide your response.",
            "structure": "xml-structured",
            "examples": True,
            "xml_tags": True,
            "prefilling": True
        }
    },
    "gemini": {
        "name": "Google Gemini",
        "rules": [
            "Four-component structure: Persona → Task → Context → Format",
            "Use chain-of-thought with self-consistency for accuracy",
            "Implement few-shot learning with consistent example formats",
            "Average 21 words for simple prompts, longer for complex tasks",
            "Define role and expertise at the beginning",
            "Provide guidance instead of giving direct orders",
            "Use natural language as if speaking to another person",
            "Include constraints to limit scope and avoid meandering"
        ],
        "enhancements": {
            "prefix": "As an expert in this domain, please help me with the following task.\n\n",
            "suffix": "\n\nPlease provide a comprehensive response while staying focused on the specific requirements.",
            "structure": "persona-task-context-format",
            "examples": True,
            "natural_language": True,
            "constraints": True
        }
    },
    "grok": {
        "name": "xAI Grok",
        "rules": [
            "Leverage real-time data capabilities for current information",
            "Use Think mode for complex reasoning and problem-solving",
            "Apply specificity and clarity principles in all prompts",
            "Implement iterative refinement for optimal results",
            "Be specific about required information sources",
            "Include ethical considerations in prompt design",
            "Use step-by-step reasoning for mathematical/logical problems",
            "Consider live data integration for trending topics"
        ],
        "enhancements": {
            "prefix": "Please approach this thoughtfully and systematically:\n\n",
            "suffix": "\n\nUse your reasoning capabilities and access to current information to provide an accurate, helpful response.",
            "structure": "clarity-specificity-reasoning",
            "examples": True,
            "real_time_data": True,
            "reasoning": True
        }
    }
}

# ===== PYDANTIC MODELS =====
class PromptRequest(BaseModel):
    """Request model for enhanced prompt processing."""
    prompt: str = Field(..., description="The original user prompt")
    target_model: str = Field(..., description="Target AI model (openai, claude, gemini, grok)")
    enhancement_type: str = Field("general", description="Type of enhancement")
    max_length: Optional[int] = Field(512, description="Maximum length for generated text")
    temperature: Optional[float] = Field(0.7, description="Generation temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    evaluate_quality: bool = Field(True, description="Whether to evaluate prompt quality")
    generate_alternatives: bool = Field(False, description="Generate multiple alternatives")
    num_alternatives: int = Field(1, description="Number of alternatives to generate")

class QualityMetrics(BaseModel):
    """Quality metrics for prompt evaluation."""
    specificity: float
    clarity: float
    completeness: float
    actionability: float
    overall: float

class GenerationMetadata(BaseModel):
    """Metadata about the generation process."""
    model: str
    processing_time: float
    temperature: float
    top_p: float
    max_length: int
    tokens_used: int

class Alternative(BaseModel):
    """Alternative prompt/generation pair."""
    prompt: str
    generated_text: str
    quality_score: float

class PromptResponse(BaseModel):
    """Enhanced response model for prompt processing results."""
    original_prompt: str
    enhanced_prompt: str
    generated_text: str
    original_quality: Optional[QualityMetrics] = None
    enhanced_quality: Optional[QualityMetrics] = None
    quality_improvement: Optional[Dict[str, float]] = None
    generation_metadata: GenerationMetadata
    processing_time: float
    alternatives: Optional[List[Alternative]] = None

class ModelInfo(BaseModel):
    """Information about available models."""
    id: str
    name: str
    description: str
    features: List[str]
    status: str

class SystemStatus(BaseModel):
    """System status response model."""
    status: str
    models_loaded: Dict[str, bool]
    available_models: List[ModelInfo]
    system_info: Dict[str, Any]
    configuration: Dict[str, Any]

# ===== FASTAPI APP INITIALIZATION =====
app = FastAPI(
    title="AI Prompt Enhancement Studio",
    description="Multi-model AI prompt enhancement system with model-specific optimization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PROMPT OPTIMIZATION ENGINE =====
class PromptOptimizer:
    """Model-specific prompt optimization engine."""
    
    @staticmethod
    def apply_model_enhancements(original_prompt: str, model_id: str, enhancement_type: str = "general") -> str:
        """Apply model-specific enhancements to a prompt."""
        if model_id not in MODEL_RULES:
            return original_prompt
        
        model = MODEL_RULES[model_id]
        enhancements = model["enhancements"]
        enhanced_prompt = original_prompt
        
        # Apply prefix and suffix
        if enhancements.get("prefix"):
            enhanced_prompt = enhancements["prefix"] + enhanced_prompt
        
        if enhancements.get("suffix"):
            enhanced_prompt = enhanced_prompt + enhancements["suffix"]
        
        # Apply model-specific formatting
        if model_id == "claude" and enhancements.get("xml_tags"):
            enhanced_prompt = f"<instructions>\n{enhanced_prompt}\n</instructions>"
        elif model_id == "gemini" and enhancements.get("constraints"):
            enhanced_prompt += "\n\nPlease stay focused on the specific requirements and provide a structured response."
        elif model_id == "grok" and enhancements.get("real_time_data"):
            enhanced_prompt = "Using current information and data, " + enhanced_prompt.lower()
        
        # Apply enhancement type specific modifications
        type_enhancements = {
            "creative_writing": "\n\nFocus on creative, engaging, and imaginative content.",
            "technical": "\n\nProvide detailed technical information with examples and implementation details.",
            "analysis": "\n\nAnalyze systematically and provide data-driven insights with clear reasoning.",
            "coding": "\n\nProvide clean, well-documented code with explanations and best practices."
        }
        
        if enhancement_type in type_enhancements:
            enhanced_prompt += type_enhancements[enhancement_type]
        
        return enhanced_prompt
    
    @staticmethod
    def evaluate_prompt_quality(prompt: str) -> QualityMetrics:
        """Evaluate prompt quality across multiple dimensions."""
        # Simple heuristic-based evaluation (replace with actual model in production)
        word_count = len(prompt.split())
        char_count = len(prompt)
        
        # Quality scoring based on prompt characteristics
        specificity = min(1.0, (word_count / 50) * 0.8 + random.uniform(0.1, 0.2))
        clarity = min(1.0, 0.7 + (0.3 if "?" in prompt or "please" in prompt.lower() else 0.1) + random.uniform(0.0, 0.2))
        completeness = min(1.0, (char_count / 200) * 0.7 + random.uniform(0.1, 0.3))
        actionability = min(1.0, 0.6 + (0.3 if any(word in prompt.lower() for word in ["create", "write", "analyze", "explain", "describe"]) else 0.1) + random.uniform(0.0, 0.2))
        overall = (specificity + clarity + completeness + actionability) / 4
        
        return QualityMetrics(
            specificity=specificity,
            clarity=clarity,
            completeness=completeness,
            actionability=actionability,
            overall=overall
        )
    
    @staticmethod
    def generate_mock_content(enhanced_prompt: str, enhancement_type: str, max_length: int = 512) -> str:
        """Generate mock content based on enhanced prompt."""
        templates = {
            "general": "This is a comprehensive response to your enhanced prompt. The system has analyzed your requirements and applied model-specific optimization techniques to provide a detailed, structured answer that addresses all key points mentioned in your request. The response demonstrates improved clarity, specificity, and actionability based on the selected AI model's best practices.",
            
            "creative_writing": "In a world where artificial intelligence and human creativity intertwine like dancers in an eternal ballet, your enhanced prompt has blossomed into a narrative that captures both technical precision and artistic expression. Each word carefully chosen, each sentence structured with the rhythm of poetry and the clarity of prose, creating a story that resonates with both logic and emotion.",
            
            "technical": """## Technical Implementation Guide

### Overview
This technical documentation provides a comprehensive implementation guide based on your enhanced prompt requirements.

### System Architecture
- **Component Design**: Modular architecture with clear separation of concerns
- **Performance Optimization**: Efficient algorithms and caching strategies
- **Security Considerations**: Input validation, authentication, and data protection
- **Scalability**: Horizontal scaling capabilities and load balancing

### Implementation Steps
1. Environment setup and dependency management
2. Core functionality implementation
3. Testing and validation procedures
4. Deployment and monitoring configuration

### Best Practices
- Follow SOLID principles for maintainable code
- Implement comprehensive error handling
- Use appropriate design patterns
- Maintain thorough documentation""",
            
            "analysis": """## Data Analysis Results

### Executive Summary
Based on the enhanced prompt analysis, the following insights have been identified through systematic evaluation and data-driven methodologies.

### Key Findings
1. **Primary Insight**: Statistical analysis reveals significant patterns in the data
2. **Correlation Analysis**: Strong relationships identified between key variables
3. **Trend Analysis**: Temporal patterns show consistent growth trajectories
4. **Risk Assessment**: Potential challenges and mitigation strategies identified

### Methodology
- Data collection and preprocessing techniques
- Statistical modeling and validation approaches
- Visualization and interpretation methods
- Quality assurance and peer review processes

### Recommendations
Based on the analysis results, the following actionable recommendations are proposed for implementation.""",
            
            "coding": """```python
# Enhanced Code Solution
# Generated based on optimized prompt requirements

class EnhancedSolution:
    \"\"\"
    Optimized implementation based on enhanced prompt specifications.
    Follows best practices for maintainability, performance, and security.
    \"\"\"
    
    def __init__(self, parameters: dict):
        self.parameters = self.validate_parameters(parameters)
        self.initialize_components()
    
    def validate_parameters(self, params: dict) -> dict:
        \"\"\"Validate input parameters with comprehensive error checking.\"\"\"
        required_keys = ['input_data', 'processing_options']
        
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")
        
        return params
    
    def initialize_components(self):
        \"\"\"Initialize system components with optimal configuration.\"\"\"
        self.processor = self.create_processor()
        self.validator = self.create_validator()
        self.output_formatter = self.create_formatter()
    
    def process_enhanced_request(self) -> dict:
        \"\"\"
        Main processing function implementing enhanced logic.
        Returns structured results with comprehensive metadata.
        \"\"\"
        try:
            # Process input with enhanced algorithms
            processed_data = self.processor.enhance_processing(
                self.parameters['input_data']
            )
            
            # Validate results
            validation_results = self.validator.validate_output(processed_data)
            
            # Format final output
            formatted_output = self.output_formatter.format_results(
                processed_data, validation_results
            )
            
            return {
                'success': True,
                'data': formatted_output,
                'metadata': self.generate_metadata()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'metadata': self.generate_error_metadata(e)
            }
    
    def generate_metadata(self) -> dict:
        \"\"\"Generate comprehensive metadata for successful processing.\"\"\"
        return {
            'processing_time': time.time(),
            'algorithm_version': '2.0.0',
            'quality_score': 0.95,
            'optimization_applied': True
        }

# Usage example
solution = EnhancedSolution(parameters={
    'input_data': 'sample_input',
    'processing_options': {'optimize': True, 'validate': True}
})

result = solution.process_enhanced_request()
print(f"Processing result: {result}")
```"""
        }
        
        content = templates.get(enhancement_type, templates["general"])
        
        # Truncate if necessary
        if len(content) > max_length:
            content = content[:max_length-3] + "..."
        
        return content

# ===== API ENDPOINTS =====
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Welcome to the AI Prompt Enhancement Studio",
        "description": "Multi-model AI prompt optimization system",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
        "models": "/models/available"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "version": "2.0.0",
        "models_available": len(MODEL_RULES)
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get detailed system status."""
    models_loaded = {model_id: True for model_id in MODEL_RULES.keys()}
    
    available_models = [
        ModelInfo(
            id=model_id,
            name=model_data["name"],
            description=f"Optimized for {model_data['name']} with {len(model_data['rules'])} optimization rules",
            features=model_data["rules"][:3],  # Show first 3 features
            status="active"
        )
        for model_id, model_data in MODEL_RULES.items()
    ]
    
    return SystemStatus(
        status="operational",
        models_loaded=models_loaded,
        available_models=available_models,
        system_info={
            "version": "2.0.0",
            "total_models": len(MODEL_RULES),
            "active_models": len(models_loaded),
            "optimization_rules": sum(len(model["rules"]) for model in MODEL_RULES.values())
        },
        configuration={
            "api_host": settings.API_HOST,
            "api_port": settings.API_PORT,
            "environment": settings.ENVIRONMENT,
            "cors_enabled": True,
            "docs_enabled": True
        }
    )

@app.get("/models/available")
async def get_available_models():
    """Get list of available AI models and their capabilities."""
    return {
        "models": [
            {
                "id": model_id,
                "name": model_data["name"],
                "description": f"AI model optimized with {len(model_data['rules'])} specific techniques",
                "rules": model_data["rules"],
                "enhancements": {
                    key: value for key, value in model_data["enhancements"].items() 
                    if isinstance(value, (str, bool))
                },
                "status": "active"
            }
            for model_id, model_data in MODEL_RULES.items()
        ],
        "total_models": len(MODEL_RULES)
    }

@app.get("/models/{model_id}/rules")
async def get_model_rules(model_id: str):
    """Get optimization rules for a specific model."""
    if model_id not in MODEL_RULES:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    model_data = MODEL_RULES[model_id]
    return {
        "model_id": model_id,
        "name": model_data["name"],
        "rules": model_data["rules"],
        "enhancements": model_data["enhancements"],
        "total_rules": len(model_data["rules"])
    }

@app.post("/enhance", response_model=PromptResponse)
async def enhance_prompt(request: PromptRequest):
    """
    Main endpoint for model-specific prompt enhancement and generation.
    
    This endpoint:
    1. Validates the target model
    2. Applies model-specific optimization rules
    3. Evaluates prompt quality (optional)
    4. Generates enhanced content
    5. Returns comprehensive results with metadata
    """
    start_time = time.time()
    
    try:
        # Validate target model
        if request.target_model not in MODEL_RULES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.target_model}. Available models: {list(MODEL_RULES.keys())}"
            )
        
        optimizer = PromptOptimizer()
        
        # Evaluate original prompt quality
        original_quality = None
        if request.evaluate_quality:
            original_quality = optimizer.evaluate_prompt_quality(request.prompt)
        
        # Apply model-specific enhancements
        enhanced_prompt = optimizer.apply_model_enhancements(
            request.prompt, 
            request.target_model, 
            request.enhancement_type
        )
        
        # Evaluate enhanced prompt quality
        enhanced_quality = None
        quality_improvement = None
        if request.evaluate_quality:
            enhanced_quality = optimizer.evaluate_prompt_quality(enhanced_prompt)
            
            # Calculate improvement metrics
            if original_quality:
                quality_improvement = {
                    "specificity": enhanced_quality.specificity - original_quality.specificity,
                    "clarity": enhanced_quality.clarity - original_quality.clarity,
                    "completeness": enhanced_quality.completeness - original_quality.completeness,
                    "actionability": enhanced_quality.actionability - original_quality.actionability,
                    "overall": enhanced_quality.overall - original_quality.overall
                }
        
        # Generate content based on enhanced prompt
        generated_text = optimizer.generate_mock_content(
            enhanced_prompt,
            request.enhancement_type,
            request.max_length
        )
        
        # Create generation metadata
        processing_time = time.time() - start_time
        generation_metadata = GenerationMetadata(
            model=MODEL_RULES[request.target_model]["name"],
            processing_time=processing_time * 1000,  # Convert to milliseconds
            temperature=request.temperature,
            top_p=request.top_p,
            max_length=request.max_length,
            tokens_used=random.randint(200, 800)  # Mock token usage
        )
        
        # Generate alternatives if requested
        alternatives = None
        if request.generate_alternatives and request.num_alternatives > 1:
            alternatives = []
            for i in range(request.num_alternatives):
                alt_prompt = f"{enhanced_prompt} [Alternative approach {i+1}]"
                alt_content = optimizer.generate_mock_content(
                    alt_prompt, 
                    request.enhancement_type, 
                    request.max_length
                )
                alt_quality = random.uniform(0.7, 0.9)
                
                alternatives.append(Alternative(
                    prompt=alt_prompt,
                    generated_text=alt_content,
                    quality_score=alt_quality
                ))
        
        # Create and return response
        return PromptResponse(
            original_prompt=request.prompt,
            enhanced_prompt=enhanced_prompt,
            generated_text=generated_text,
            original_quality=original_quality,
            enhanced_quality=enhanced_quality,
            quality_improvement=quality_improvement,
            generation_metadata=generation_metadata,
            processing_time=processing_time,
            alternatives=alternatives
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@app.post("/batch-enhance")
async def batch_enhance_prompts(prompts: List[str], target_model: str, enhancement_type: str = "general"):
    """Batch process multiple prompts for the same model."""
    if target_model not in MODEL_RULES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported model: {target_model}"
        )
    
    start_time = time.time()
    results = []
    optimizer = PromptOptimizer()
    
    for prompt in prompts:
        enhanced_prompt = optimizer.apply_model_enhancements(prompt, target_model, enhancement_type)
        generated_text = optimizer.generate_mock_content(enhanced_prompt, enhancement_type)
        
        results.append({
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "generated_text": generated_text,
            "quality_score": random.uniform(0.7, 0.95)
        })
    
    processing_time = time.time() - start_time
    
    return {
        "results": results,
        "total_prompts": len(prompts),
        "target_model": MODEL_RULES[target_model]["name"],
        "processing_time": processing_time,
        "average_time_per_prompt": processing_time / len(prompts) if prompts else 0
    }

@app.get("/evaluate/{prompt}")
async def evaluate_single_prompt(prompt: str):
    """Evaluate a single prompt's quality metrics."""
    optimizer = PromptOptimizer()
    quality_metrics = optimizer.evaluate_prompt_quality(prompt)
    
    return {
        "prompt": prompt,
        "quality_metrics": quality_metrics,
        "evaluation_timestamp": time.time(),
        "prompt_length": len(prompt),
        "word_count": len(prompt.split())
    }

if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
