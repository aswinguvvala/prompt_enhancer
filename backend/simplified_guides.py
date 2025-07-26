# COMPREHENSIVE TEXT-BASED MODEL ENHANCEMENT RULES
# Model-specific prompt engineering principles without XML formatting

SIMPLIFIED_MODEL_GUIDES = {
    "claude": {
        "name": "Anthropic Claude",
        "description": "Constitutional AI model optimized for helpful, harmless, and honest responses",
        "rules": [
            "Be clear and direct with explicit, unambiguous instructions",
            "Provide specific examples when requesting structured output or format",
            "Assign a specific role or expertise (e.g. 'You are an expert legal analyst')",
            "Separate instructions from data clearly using natural language transitions",
            "Ask Claude to think step-by-step for complex reasoning tasks",
            "Use conversational, thoughtful language that works with Claude's alignment",
            "Provide context and background information for better understanding",
            "Structure requests with clear task definition and desired outcome",
            "For multi-step tasks, break them into clear sequential instructions",
            "Use encouraging, collaborative language rather than commanding tone"
        ],
        "output_guidelines": [
            "Request natural language responses without technical markup",
            "Ask for step-by-step explanations in conversational format",
            "Specify desired format using natural language descriptions",
            "Encourage detailed, thoughtful responses with reasoning"
        ],
        "avoid": [
            "Never use XML tags, angle brackets, or structured markup",
            "Avoid technical formatting instructions",
            "Don't use programming-style syntax in prompts",
            "Avoid overly complex nested instructions"
        ]
    },
    "openai": {
        "name": "OpenAI GPT-4",
        "description": "Advanced reasoning model optimized for complex instructions and tool use",
        "rules": [
            "Write clear, detailed instructions with specific context",
            "Use explicit role assignment (e.g. 'Act as an expert copywriter')",
            "Provide examples to demonstrate desired output format",
            "Break complex tasks into simple, sequential subtasks",
            "Ask the model to work out solutions step-by-step before concluding",
            "Specify the desired length and format of responses",
            "Use delimiters like quotes or dashes to separate different sections",
            "Provide reference text when accuracy is crucial",
            "Give the model time to think through problems methodically",
            "Structure prompts with clear beginning, task, and completion criteria"
        ],
        "output_guidelines": [
            "Request structured responses using natural language descriptions",
            "Ask for reasoning explanations before final answers",
            "Specify output format preferences in plain language",
            "Encourage comprehensive, well-organized responses"
        ],
        "avoid": [
            "Don't use XML tags or technical markup syntax",
            "Avoid vague or ambiguous instructions",
            "Don't rush the model to conclusions without reasoning",
            "Avoid overly complex single-prompt requests"
        ]
    },
    "gemini": {
        "name": "Google Gemini",
        "description": "Multimodal AI model with native image, text, and reasoning capabilities",
        "rules": [
            "Define a clear persona or role for the AI to adopt",
            "State the specific task with detailed context and requirements",
            "Provide relevant background information and constraints",
            "Specify the desired output format using descriptive language",
            "Use structured approach: persona, task, context, format",
            "Encourage systematic, methodical responses",
            "Request clear explanations with relevant examples",
            "Ask for step-by-step breakdowns of complex processes",
            "Use iterative conversation for refinement and improvement",
            "Provide specific criteria for evaluation and success"
        ],
        "output_guidelines": [
            "Ask for well-structured, organized responses",
            "Request practical, actionable insights and recommendations",
            "Encourage comprehensive coverage of the topic",
            "Specify preference for clear, accessible language"
        ],
        "avoid": [
            "Don't use technical markup or XML-style formatting",
            "Avoid overly broad or unfocused requests",
            "Don't skip providing necessary context",
            "Avoid requesting multiple unrelated tasks simultaneously"
        ]
    },
    "grok": {
        "name": "xAI Grok",
        "description": "Real-time data integration model with direct, practical reasoning",
        "rules": [
            "Be direct and practical in your requests",
            "Ask for specific information with clear parameters",
            "Request step-by-step reasoning for complex problems",
            "Leverage real-time data by asking for current information",
            "Use clear, straightforward language without unnecessary complexity",
            "Ask for evidence-based responses with source consideration",
            "Request practical, actionable insights and recommendations",
            "Be specific about the type of analysis or information needed",
            "Ask for balanced perspectives on controversial topics",
            "Request fact-checking and verification when appropriate"
        ],
        "output_guidelines": [
            "Ask for direct, practical responses without fluff",
            "Request current, up-to-date information when relevant",
            "Encourage critical analysis and evidence-based reasoning",
            "Specify preference for clear, actionable conclusions"
        ],
        "avoid": [
            "Don't use complex formatting or markup syntax",
            "Avoid overly academic or theoretical language",
            "Don't ask for speculation without acknowledging uncertainty",
            "Avoid requests that ignore the model's real-time capabilities"
        ]
    }
}

# ANTI-XML INSTRUCTIONS FOR ALL MODELS
ANTI_XML_INSTRUCTIONS = [
    "IMPORTANT: Provide your enhanced prompt using natural language only",
    "Do not use XML tags, angle brackets, or any technical markup",
    "Avoid structured formatting like <thinking>, <example>, <instructions>, etc.",
    "Write in clear, natural prose without programming-style syntax",
    "Use regular punctuation and paragraph structure for organization",
    "Present information in conversational, readable format"
]

# ENHANCEMENT TEMPLATE INSTRUCTIONS
ENHANCEMENT_INSTRUCTIONS = [
    "Study the original prompt and identify areas for improvement",
    "Apply model-specific optimization principles provided above",
    "Enhance clarity, specificity, and effectiveness",
    "Maintain the user's original intent and requirements",
    "Add helpful context, examples, or structure as needed",
    "Create a more effective version that will produce better AI responses"
]