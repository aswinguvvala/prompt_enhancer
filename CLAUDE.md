# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The **AI Prompt Enhancement Studio** is a real AI-powered prompt optimization system that uses free/local models (Ollama, Hugging Face Transformers, etc.) to enhance user prompts with model-specific best practices for OpenAI GPT-4, Anthropic Claude, Google Gemini, and xAI Grok. The system feeds comprehensive, text-based prompt engineering rules into the AI model's context window along with the user's original prompt to generate natural language enhanced prompts without XML tags or technical markup.

## Development Commands

### Environment Setup & Dependencies
```bash
# Navigate to project directory
cd prompt-enhancement-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (includes transformers, torch, etc.)
pip install -r requirements.txt

# For Ollama integration (alternative)
# Install Ollama: https://ollama.ai/
# ollama pull llama2  # or preferred model
```

### Running the Application
```bash
# Quick start (recommended)
python start_server.py

# Manual backend start
cd backend
uvicorn app:app --host localhost --port 8000 --reload

# Open frontend (manual)
open frontend/index.html  # macOS
```

### Testing
```bash
# Run comprehensive integration tests
python test_system.py

# Tests validate:
# - Model rules configuration and loading
# - AI model integration and response generation
# - Prompt enhancement pipeline functionality
# - Model-specific rule application
# - Error handling and fallbacks
```

## Architecture Overview

### Real AI Model Integration Architecture

**AI Model Layer**: Integration with free/local AI models:
- **Ollama Integration**: Local model serving with API calls
- **Hugging Face Transformers**: Direct model loading and inference
- **OpenAI-Compatible APIs**: Support for local deployments (LM Studio, etc.)

**Rule-Based Enhancement Engine**: Feeds model-specific optimization rules into AI model context along with user prompt to generate enhanced versions.

**Frontend Layer** (`frontend/`): Static HTML/CSS/JavaScript interface communicating with FastAPI backend.

**Backend Layer** (`backend/`): FastAPI application orchestrating AI model calls with rule injection.

### Core Components

#### AI Model Integration (`backend/models/`)
- **prompt_enhancer.py**: Handles communication with local AI models (Ollama, Transformers)
- **main_llm.py**: Manages model loading, context preparation, and inference
- **evaluator.py**: Removed outdated quality metrics, replaced with AI model validation

#### Model Rules System (`backend/app.py`)
- **MODEL_RULES**: Comprehensive rule sets for each AI model (OpenAI, Claude, Gemini, Grok)
- **Rule Injection**: Rules are fed into the AI model's context window along with user prompt
- **Context Template**: Structured prompt template that instructs the AI model how to enhance prompts

#### Enhancement Pipeline
1. **User Input**: Original prompt + target model selection
2. **Rule Loading**: Retrieve model-specific optimization rules
3. **Context Preparation**: Combine rules + user prompt + enhancement instructions
4. **AI Model Call**: Send to local AI model (Ollama/Transformers/etc.)
5. **Response Processing**: Extract and validate enhanced prompt
6. **Content Generation**: Optional - use enhanced prompt for content generation

### Free AI Model Options

#### Primary Integration: Ollama
```bash
# Install and setup Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2  # or mistral, codellama, etc.
ollama serve
```

#### Alternative Options:
- **Hugging Face Transformers**: Direct model loading (Llama2, Mistral, CodeLlama)
- **LM Studio**: Local OpenAI-compatible API server
- **GPT4All**: Local model serving
- **Oobabooga Text Generation WebUI**: Advanced local model interface

### Text-Based Enhancement System (Anti-XML Approach)

The system uses a comprehensive **text-based enhancement approach** that explicitly avoids XML tags and technical markup. This ensures enhanced prompts use natural language optimization techniques.

#### Key Features
- **Comprehensive Rule Injection**: Detailed model-specific prompt engineering rules fed into AI context
- **Anti-XML Instructions**: Explicit instructions to prevent XML tags like `<thinking>`, `<example>`, etc.
- **Natural Language Focus**: Emphasizes prose-based enhancement techniques
- **Model-Specific Optimization**: Tailored rules for each target AI model

#### Rule Structure (`backend/simplified_guides.py`)
Each model has comprehensive rules organized into:
- **Rules**: Core optimization principles for the target model
- **Output Guidelines**: Preferred response format and style
- **Avoid**: What not to do (including XML tags and technical markup)

#### Anti-XML System
The system includes explicit anti-XML instructions:
- "IMPORTANT: Provide your enhanced prompt using natural language only"
- "Do not use XML tags, angle brackets, or any technical markup"
- "Avoid structured formatting like `<thinking>`, `<example>`, `<instructions>`, etc."
- "Write in clear, natural prose without programming-style syntax"

### Context Window Strategy

#### Comprehensive Rule Injection Template
```
You are an expert prompt engineer specializing in [TARGET_MODEL] optimization.

TARGET MODEL: [Model Name]
DESCRIPTION: [Model characteristics and capabilities]

OPTIMIZATION RULES FOR [TARGET_MODEL]:
- [Comprehensive list of model-specific rules]

OUTPUT GUIDELINES:
- [Preferred response format and style guidelines]

IMPORTANT - WHAT TO AVOID:
- [Model-specific things to avoid, including XML tags]

CRITICAL - NO TECHNICAL MARKUP:
- IMPORTANT: Provide your enhanced prompt using natural language only
- Do not use XML tags, angle brackets, or any technical markup
- [Additional anti-XML instructions]

ENHANCEMENT PROCESS:
- Study the original prompt and identify areas for improvement
- Apply model-specific optimization principles provided above
- [Additional enhancement instructions]

ORIGINAL PROMPT TO ENHANCE:
"[User's original prompt]"

Please create an enhanced version that follows all the [TARGET_MODEL] optimization rules above. Write your enhanced prompt in natural language without any XML tags, technical markup, or structured formatting.

ENHANCED PROMPT:
```

## Development Patterns

### AI Model Integration Pattern
```python
class PromptEnhancer:
    def __init__(self, model_type="ollama"):  # or "transformers", "openai_compatible"
        self.model_client = self._initialize_model(model_type)
    
    async def enhance_prompt(self, original_prompt, target_model, enhancement_type):
        # Load model-specific rules
        rules = MODEL_RULES[target_model]["rules"]
        
        # Prepare context with rules + prompt
        context = self._prepare_context(rules, original_prompt, target_model, enhancement_type)
        
        # Call AI model
        enhanced_prompt = await self.model_client.generate(context)
        
        return enhanced_prompt
```

### Configuration Management (`backend/config.py`)
Extended settings for AI model integration:
- **Model Selection**: Ollama, Transformers, OpenAI-compatible API
- **Model Parameters**: Temperature, max_tokens, model_name
- **API Endpoints**: Ollama server URL, Hugging Face model paths
- **Performance Settings**: Batch size, timeout, concurrent requests

### Error Handling & Fallbacks
- **Model Unavailable**: Fallback to alternative local models
- **Context Window Exceeded**: Rule truncation and smart compression
- **Generation Failures**: Retry logic with simplified context
- **Network Issues**: Local model fallback chains

## Implementation Notes

### Real AI Model Requirements
- **No Mock Data**: All responses generated by actual AI models
- **Local/Free Models**: Ollama, Hugging Face, or other free alternatives
- **Rule-Based Enhancement**: Model rules fed into AI context for optimization
- **Removed Quality Metrics**: No hardcoded specificity/clarity scoring

### Model Rules Integration (Text-Based)
Each model's rules are dynamically injected into the AI model's context using natural language:
- **OpenAI GPT-4 Rules**: Clear instructions, role assignment, step-by-step reasoning, natural delimiters
- **Anthropic Claude Rules**: Conversational language, clear task definition, thoughtful prompting (NO XML tags)
- **Google Gemini Rules**: Persona-task-context-format structure in natural language
- **xAI Grok Rules**: Direct practical language, evidence-based requests, current information focus

### Anti-XML Implementation
The system actively prevents XML tag generation through:

#### Rule-Level Prevention
- Each model's "avoid" rules explicitly mention XML tags and technical markup
- Natural language alternatives provided for every XML-based technique
- Emphasis on conversational, prose-based enhancement

#### Context-Level Instructions
- Multiple layers of anti-XML instructions in the AI context
- Explicit prohibition of angle brackets, technical symbols, structured markup
- Clear guidance on natural language formatting

#### Response Processing
- Enhanced prompt extraction focuses on natural language content
- Filters out any residual technical formatting that might appear
- Validates enhanced prompts against anti-XML criteria

### Performance Considerations
- **Model Loading**: Cache loaded models in memory
- **Context Optimization**: Efficient rule injection without exceeding context limits
- **Async Processing**: Non-blocking AI model calls
- **Batch Support**: Multiple prompt enhancement in parallel

### Extension Points
- **New AI Models**: Add support for additional local model servers
- **Custom Rules**: User-defined optimization rules for specialized models
- **Model Chaining**: Use multiple AI models in enhancement pipeline
- **Rule Learning**: Dynamic rule adjustment based on enhancement success