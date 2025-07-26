# SIMPLIFIED MODEL GUIDES FOR FASTER PROCESSING WITH 3B MODELS
# Optimized for speed - reduced from comprehensive guides to essential rules only

SIMPLIFIED_MODEL_GUIDES = {
    "claude": {
        "name": "Anthropic Claude",
        "core_principles": [
            "Use XML tags for structure: <instructions>, <thinking>, <examples>",
            "Let Claude think step-by-step in <thinking> tags", 
            "Work with Claude's values, frame requests positively"
        ],
        "essential_techniques": {
            "structure": "Use <instructions>task</instructions> format",
            "reasoning": "Request <thinking>step-by-step reasoning</thinking>", 
            "examples": "Provide <examples><example>input→output</example></examples>"
        },
        "advanced_features": []
    },
    "openai": {
        "name": "OpenAI GPT-4",
        "core_principles": [
            "Write clear, explicit instructions with delimiters",
            "Break complex tasks into simpler subtasks",
            "Ask model to think step-by-step before answering"
        ],
        "essential_techniques": {
            "delimiters": "Use ###, ```, or XML tags to separate sections",
            "systematic": "Ask to plan approach step-by-step",
            "role": "Set clear system role and persona"
        },
        "advanced_features": []
    },
    "gemini": {
        "name": "Google Gemini",
        "core_principles": [
            "Use Persona-Task-Context-Format (PTCF) structure",
            "Design for iterative conversation", 
            "Specify exact output format"
        ],
        "essential_techniques": {
            "ptcf": "Persona: [role], Task: [action], Context: [info], Format: [output]",
            "iterative": "Design for follow-up refinement",
            "format": "Specify exact output requirements"
        },
        "advanced_features": []
    },
    "grok": {
        "name": "xAI Grok",
        "core_principles": [
            "Leverage real-time web data access",
            "Use appropriate mode: Standard/Fun/Think",
            "Apply critical thinking over popular narratives"
        ],
        "essential_techniques": {
            "modes": "Choose Standard (facts), Fun (witty), Think (reasoning)",
            "research": "Frame as research tasks for live data",
            "sources": "Request citations and source examination"
        },
        "advanced_features": []
    }
}