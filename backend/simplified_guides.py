# COMPREHENSIVE 2024-2025 AI MODEL ENHANCEMENT RULES
# Research-backed model-specific prompt engineering principles from official documentation

SIMPLIFIED_MODEL_GUIDES = {
    "claude": {
        "name": "Anthropic Claude 4 (Opus/Sonnet)",
        "description": "Constitutional AI model with advanced reasoning, XML structure support, and multishot prompting capabilities",
        "rules": [
            # Core 2024-2025 Best Practices from Official Anthropic Documentation
            "Be clear and explicit with instructions - Claude 4 responds well to direct, unambiguous guidance",
            "Use XML tags for structure: <example>, <document>, <thinking>, <instructions> to organize your prompt",
            "Implement multishot prompting with 3-5 diverse, relevant examples showing exactly what you want",
            "Include 'Think step by step' to give Claude time to reason through complex problems before responding",
            "Use human messages for main instructions rather than system messages - Claude follows human prompts better",
            "Provide comprehensive context and background information - Claude doesn't retain info between conversations",
            "Structure prompts in recommended order: Task context → Tone context → Background data → Detailed task",
            "Use conversational, collaborative language that works with Claude's alignment ('Could you please' vs commands)",
            "Break complex tasks into smaller, sequential steps with clear dependencies and logical flow",
            "Assign specific roles or expertise ('You are an expert legal analyst specializing in corporate law')",
            "Include success criteria and evaluation metrics to define what constitutes a good response",
            "Use prefilling to guide output format by starting Claude's response with your desired structure",
            "Request step-by-step reasoning and explanations for transparency in complex problem-solving",
            "Avoid negative prompting (telling Claude what NOT to do) as it can backfire through reverse psychology",
            "Leverage Claude's broad knowledge by providing relevant domain-specific context and technical details",
            "Use iterative refinement - build on responses through follow-up questions and adjustments",
            # Advanced 2024 Techniques
            "Apply chain-of-thought reasoning by explicitly requesting detailed thought processes before conclusions",
            "Use prompt scaffolding with structured templates that guide Claude's thinking and response format",
            "Include emotional context when relevant ('This is important to my career') to improve response quality",
            # Latest 2025 Features
            "For complex analysis, request Claude to consider multiple perspectives and potential counterarguments",
            "Use constitutional AI principles by asking Claude to consider helpfulness, harmlessness, and honesty"
        ],
        "output_guidelines": [
            "Request responses using natural language structure without technical markup",
            "Ask for detailed explanations with clear reasoning chains and logical progression",
            "Specify desired format using conversational descriptions rather than code-like formatting",
            "Encourage comprehensive, thoughtful responses with multiple angles and considerations",
            "Request structured organization using headings, bullet points, and numbered lists in natural language",
            "Ask for citations and references when Claude uses specific information or makes factual claims",
            "Encourage collaborative tone and respectful, helpful language throughout responses",
            "Request examples and practical applications to illustrate abstract concepts"
        ],
        "avoid": [
            "Never use excessive XML tags - use them strategically for structure only",
            "Avoid commanding or directive language - prefer collaborative, respectful requests",
            "Don't over-rely on negative prompting - focus on what you DO want rather than what you don't",
            "Avoid overly complex nested instructions that could confuse the reasoning process",
            "Don't assume context from previous conversations - always provide necessary background",
            "Avoid rushing Claude to conclusions - allow space for thoughtful, step-by-step analysis",
            "Don't skip providing clear examples when requesting specific formats or styles of response"
        ]
    },
    "openai": {
        "name": "OpenAI GPT-4o/GPT-4.1",
        "description": "Advanced reasoning model with enhanced instruction-following, 1M token context, and systematic problem-solving capabilities",
        "rules": [
            # Official OpenAI 2024-2025 Best Practices (6 Core Strategies)
            "Write clear, specific instructions with comprehensive context - be explicit about requirements and constraints",
            "Provide reference text when accuracy is crucial - include relevant documents, data, or source material",
            "Split complex tasks into simpler subtasks with clear sequential steps and dependencies",
            "Give the model time to think - request step-by-step reasoning before providing final answers",
            "Use external tools and functions when appropriate for calculations, lookups, or specialized tasks",
            "Test changes systematically by comparing different prompt variations and measuring results",
            # Advanced Instruction Following (GPT-4.1 Specific)
            "Use explicit role assignment with detailed expertise ('Act as a senior data scientist with 10+ years experience')",
            "Implement few-shot prompting with 1-3 high-quality examples showing desired input/output patterns",
            "Structure prompts with clear delimiters (###, triple quotes, or distinct sections) to separate information types",
            "Apply iterative refinement approach - use multiple rounds of interaction for complex tasks",
            "Include specific length, format, and structure requirements for responses",
            "Use system messages for persona, core directives, and overall behavior guidelines",
            "Request JSON or structured formats when you need machine-readable output",
            "Leverage chain-of-thought prompting by asking for detailed reasoning processes",
            # Context Window Optimization (1M tokens)
            "For long documents, structure information hierarchically with clear sections and summaries",
            "Use the full context window strategically for comprehensive analysis and multi-hop reasoning",
            "Include persistent reminders and key constraints throughout long prompts",
            # 2025 Performance Enhancements
            "Add emotional stimuli phrases ('This is very important to my career') to increase response quality",
            "Use prompt scaffolding with defensive structures to prevent misuse and ensure appropriate responses",
            "Apply self-consistency techniques by requesting multiple reasoning paths for verification",
            "Include uncertainty acknowledgment - allow the model to express confidence levels and limitations"
        ],
        "output_guidelines": [
            "Request structured responses with clear hierarchical organization and logical flow",
            "Ask for detailed reasoning explanations before final conclusions or recommendations",
            "Specify output format preferences using natural language descriptions with examples",
            "Encourage comprehensive responses with practical examples and real-world applications",
            "Request numbered steps for processes and bullet points for features, benefits, or key information",
            "Ask for confidence levels and uncertainty acknowledgment when appropriate",
            "Encourage use of headers, sections, and clear organizational structure for readability",
            "Request citations, sources, and evidence when making factual claims or recommendations"
        ],
        "avoid": [
            "Don't use vague or ambiguous instructions - be specific about all requirements and expectations",
            "Avoid rushing to conclusions without allowing time for systematic reasoning and analysis",
            "Don't mix different types of information without clear delimiters or structural separation",
            "Avoid providing low-quality or inconsistent examples that could confuse the learning process",
            "Don't skip providing reference materials when accuracy and factual correctness are crucial",
            "Avoid overly complex single-prompt requests - break them into manageable phases or iterations",
            "Don't ignore the model's reasoning capabilities - leverage step-by-step thinking for better results"
        ]
    },
    "gemini": {
        "name": "Google Gemini 1.5 Pro (2025)",
        "description": "Multimodal AI with 1M+ token context, PTCF framework optimization, and advanced reasoning capabilities",
        "rules": [
            # Official Google 2024-2025 PTCF Framework
            "Use PTCF structure: Persona → Task → Context → Format for optimal results and consistency",
            "Define clear persona with specific expertise ('As a market research analyst with 5+ years experience')",
            "State tasks explicitly with detailed requirements and measurable success criteria",
            "Provide comprehensive context including background, constraints, and audience information",
            "Specify exact output format using descriptive language and structural requirements",
            # Core Google Best Practices
            "Keep prompts under 4,000 characters for optimal performance and response quality",
            "Use few-shot examples carefully - 1-3 examples max to avoid overfitting and maintain flexibility",
            "Write conversationally as if explaining to a knowledgeable colleague or professional peer",
            "Include precise, specific details and avoid ambiguous language or vague requirements",
            "Break complex problems into multiple focused requests for better accuracy and clarity",
            # Advanced 2025 Techniques  
            "Apply chain-of-thought with self-consistency for improved accuracy on complex reasoning tasks",
            "Use iterative conversation for refinement - build on responses through follow-up questions",
            "Experiment with content order as position can significantly affect response quality",
            "Combine multiple prompting approaches (few-shot + chain-of-thought + role-playing) strategically",
            "Include necessary context rather than assuming the model has all required background information",
            # Multimodal Capabilities (2025)
            "For multimodal tasks, clearly specify how text and visual elements should be integrated",
            "Use descriptive language for image analysis and generation requirements",
            "Structure multimodal prompts with clear sections for different input types",
            # Performance Optimization
            "Use guidance language rather than commands - frame requests as collaborative problem-solving",
            "Front-load important information early in prompts for better attention and processing",
            "Add emotional context and stakes when relevant to improve response engagement and quality",
            "Request self-reflection and error-checking for critical tasks requiring high accuracy"
        ],
        "output_guidelines": [
            "Request well-structured, hierarchically organized responses with clear sections and flow",
            "Ask for practical, actionable insights with specific next steps and implementation guidance",
            "Encourage comprehensive topic coverage with relevant details and supporting information",
            "Specify language preferences - accessible but thorough, appropriate for target audience",
            "Request numbered processes, bulleted key points, and tabular data when appropriate",
            "Ask for examples, case studies, and real-world applications to illustrate concepts",
            "Encourage natural language without technical markup or programming-style formatting",
            "Request confidence indicators and acknowledgment of limitations when dealing with uncertain topics"
        ],
        "avoid": [
            "Don't use technical markup, XML-style formatting, or programming syntax in requests",
            "Avoid overly broad or unfocused requests - be specific about scope and desired outcomes",
            "Don't skip providing necessary context and background information for complex topics",
            "Avoid requesting multiple unrelated tasks simultaneously - maintain clear focus",
            "Don't ignore audience considerations - specify who will use the output",
            "Avoid poor format specification - clearly define desired output structure and organization",
            "Don't provide insufficient or confusing examples when requesting specific styles or formats"
        ]
    },
    "grok": {
        "name": "xAI Grok 3/4 with Think Mode",
        "description": "Real-time AI with Think Mode reasoning, DeepSearch capabilities, and current information access",
        "rules": [
            # Core xAI 2024-2025 Capabilities
            "Be direct and practical - use clear, straightforward language without unnecessary complexity",
            "Leverage Think Mode for complex reasoning by explicitly requesting step-by-step problem-solving",
            "Use DeepSearch capabilities for current information by asking for recent data and trending topics",
            "Request evidence-based responses with source consideration and real-time fact-checking",
            "Ask for practical, actionable insights with real-world applications and immediate relevance",
            # Think Mode Optimization (2025)
            "For complex problems, request 'Think Mode' analysis with detailed reasoning chains",
            "Use step-by-step problem decomposition for mathematical, logical, or analytical tasks",
            "Ask for multiple solution approaches and comparative analysis when appropriate",
            "Request confidence levels and uncertainty acknowledgment for complex reasoning tasks",
            # Real-Time Data Integration
            "Specify when you need current, up-to-date information vs. general knowledge",
            "Ask for trending topics, recent developments, and real-time analysis when relevant",
            "Request verification of current facts and cross-referencing with multiple sources",
            "Use Grok's X/Twitter integration for social media trends and public sentiment analysis",
            # Communication Style
            "Use straightforward, conversational language - avoid academic jargon or overpowering complexity",
            "Be specific about information requirements, scope, and desired level of detail",
            "Request balanced perspectives on controversial topics with multiple viewpoints represented",
            "Ask for practical examples and real-world applications rather than purely theoretical discussions",
            # Advanced Features (2024-2025)
            "Include sufficient background context for complex or niche topics requiring specialized knowledge",
            "Request structured responses with bullet points, numbered steps, or clear organizational hierarchy",
            "Use iterative refinement - build on responses and add specificity if initial answers are too broad",
            "Leverage Grok's analytical capabilities for data interpretation and pattern recognition",
            "Ask for critical analysis and evidence-based reasoning with supporting documentation"
        ],
        "output_guidelines": [
            "Request direct, practical responses without unnecessary verbosity or academic flourishes",
            "Ask for current, up-to-date information when relevant, leveraging real-time data capabilities",
            "Encourage critical analysis with evidence-based reasoning and multiple source verification",
            "Specify preference for clear, actionable conclusions with practical next steps",
            "Request bullet points for key information and numbered steps for processes and procedures",
            "Ask for balanced viewpoints on complex topics with acknowledgment of different perspectives",
            "Encourage clear headings and structured organization for longer responses",
            "Request source citations and evidence references when making factual claims or recommendations"
        ],
        "avoid": [
            "Don't use overly complex formatting or markup syntax - keep requests straightforward",
            "Avoid purely academic or theoretical language without practical application or relevance",
            "Don't ask for speculation without acknowledging uncertainty appropriately",
            "Avoid ignoring Grok's real-time capabilities when current information would be valuable",
            "Don't be overly general or unfocused - provide specific parameters and requirements",
            "Avoid missing context for complex topics - provide sufficient background information",
            "Don't ignore Grok's analytical strengths for data interpretation and critical reasoning tasks"
        ]
    }
}

# UNIVERSAL 2025 PROMPTING PRINCIPLES
UNIVERSAL_2025_PRINCIPLES = [
    "Use emotional stimuli phrases ('This is very important to my career') to increase response quality by 10-15%",
    "Apply defensive prompting with structured templates to prevent misuse and ensure appropriate responses",
    "Include 'take a deep breath and work on this problem step-by-step' for complex reasoning tasks",
    "Use multimodal capabilities when available - combine text, images, and structured data strategically",
    "Request uncertainty acknowledgment and confidence levels for critical decisions",
    "Apply prompt scaffolding with clear templates that guide thinking and response structure",
    "Use iterative refinement approach - build complexity through multiple interaction rounds",
    "Include real-time optimization cues when models support current information access"
]

# ANTI-XML INSTRUCTIONS (Updated for 2025)
ANTI_XML_INSTRUCTIONS = [
    "CRITICAL: Provide enhanced prompts using natural language only - no technical markup",  
    "Do not use XML tags, angle brackets, or any programming-style formatting",
    "Avoid structured markup like <thinking>, <example>, <instructions> unless specifically for Claude",
    "Write in clear, conversational prose with natural paragraph structure and organization",
    "Use standard punctuation, headers, and bullet points for organization - no code syntax",
    "Present information in professional, readable format appropriate for human communication"
]

# COMPREHENSIVE 2025 ENHANCEMENT INSTRUCTIONS
ENHANCEMENT_INSTRUCTIONS = [
    "Analyze the original prompt using 2024-2025 best practices for the target AI model",
    "Apply model-specific optimization techniques systematically based on official documentation",
    "Enhance clarity, specificity, and structure while maintaining the user's core intent and requirements",
    "Add appropriate context, examples, and background information optimized for the target model",
    "Incorporate advanced techniques: chain-of-thought reasoning, few-shot examples, role assignment",
    "Include emotional context and stakes when relevant to improve response engagement",
    "Structure the enhanced prompt using the model's preferred organization and communication style",
    "Add success criteria, output format specifications, and quality guidelines",
    "Include defensive prompting elements to ensure appropriate, helpful responses",
    "Optimize for the model's latest capabilities: multimodal processing, real-time data, extended context"
]

# MODEL PERFORMANCE OPTIMIZATION MATRIX (2025)
MODEL_CAPABILITIES = {
    "claude": {
        "context_window": "200K tokens",
        "specialties": ["Reasoning", "Analysis", "Writing", "Code"],
        "latest_features": ["Constitutional AI", "XML Structure", "Multishot Prompting"],
        "optimization_focus": "Collaborative reasoning with structured thinking"
    },
    "openai": {
        "context_window": "128K tokens (GPT-4o), 1M tokens (GPT-4.1)",
        "specialties": ["Reasoning", "Tool Use", "Structured Output", "Code"],
        "latest_features": ["Advanced Reasoning", "Function Calling", "Vision"],
        "optimization_focus": "Systematic problem-solving with clear instructions"
    },
    "gemini": {
        "context_window": "1M+ tokens",
        "specialties": ["Multimodal", "Analysis", "Code", "Research"],
        "latest_features": ["PTCF Framework", "Multimodal Integration", "Long Context"],
        "optimization_focus": "Structured analysis with comprehensive context"
    },
    "grok": {
        "context_window": "128K tokens",
        "specialties": ["Real-time Data", "Analysis", "Current Events", "Think Mode"],
        "latest_features": ["Think Mode", "DeepSearch", "X Integration", "Real-time Access"],
        "optimization_focus": "Practical reasoning with current information"
    }
}