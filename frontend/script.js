// ===== AI PROMPT ENHANCEMENT STUDIO - JAVASCRIPT =====

// ===== GLOBAL STATE =====
let selectedModel = null;
let currentTheme = 'light';
let isProcessing = false;

// ===== MODEL-SPECIFIC OPTIMIZATION RULES =====
const MODEL_RULES = {
    openai: {
        name: 'OpenAI GPT-4',
        rules: [
            'Be explicit and specific with clear instructions',
            'Use iterative refinement approach for complex tasks',
            'Structure prompts with delimiters and format preferences', 
            'Implement chain-of-thought reasoning with "Think step by step"',
            'Consider recency bias - place important info at the end',
            'Include specific examples to show desired patterns',
            'Specify output format explicitly (JSON, markdown, etc.)',
            'Use role definition at the beginning of prompts'
        ],
        enhancements: {
            prefix: 'You are an expert assistant. Please approach this systematically:\n\n',
            suffix: '\n\nPlease think step by step and provide a detailed, well-structured response.',
            structure: 'role-context-task-format',
            examples: true,
            chainOfThought: true
        }
    },
    claude: {
        name: 'Anthropic Claude',
        rules: [
            'Use XML tags for structure (<example>, <document>, <thinking>)',
            'Implement multishot prompting with 3-5 diverse examples',
            'Include chain-of-thought reasoning before final answers',
            'Use prefilling to guide output format',
            'Provide context and motivation behind instructions',
            'Break complex tasks into smaller subtasks',
            'Structure: Task context ’ Tone ’ Background ’ Detailed task',
            'Use explicit instructions with clear explanations'
        ],
        enhancements: {
            prefix: '<task>\nYou are an expert assistant who thinks carefully before responding.\n',
            suffix: '\n</task>\n\nPlease think through this step by step in <thinking> tags, then provide your response.',
            structure: 'xml-structured',
            examples: true,
            xmlTags: true,
            prefilling: true
        }
    },
    gemini: {
        name: 'Google Gemini',
        rules: [
            'Four-component structure: Persona ’ Task ’ Context ’ Format',
            'Use chain-of-thought with self-consistency for accuracy',
            'Implement few-shot learning with consistent example formats',
            'Average 21 words for simple prompts, longer for complex tasks',
            'Define role and expertise at the beginning',
            'Provide guidance instead of giving direct orders',
            'Use natural language as if speaking to another person',
            'Include constraints to limit scope and avoid meandering'
        ],
        enhancements: {
            prefix: 'As an expert in this domain, please help me with the following task.\n\n',
            suffix: '\n\nPlease provide a comprehensive response while staying focused on the specific requirements.',
            structure: 'persona-task-context-format',
            examples: true,
            naturalLanguage: true,
            constraints: true
        }
    },
    grok: {
        name: 'xAI Grok',
        rules: [
            'Leverage real-time data capabilities for current information',
            'Use Think mode for complex reasoning and problem-solving',
            'Apply specificity and clarity principles in all prompts',
            'Implement iterative refinement for optimal results',
            'Be specific about required information sources',
            'Include ethical considerations in prompt design',
            'Use step-by-step reasoning for mathematical/logical problems',
            'Consider live data integration for trending topics'
        ],
        enhancements: {
            prefix: 'Please approach this thoughtfully and systematically:\n\n',
            suffix: '\n\nUse your reasoning capabilities and access to current information to provide an accurate, helpful response.',
            structure: 'clarity-specificity-reasoning',
            examples: true,
            realTimeData: true,
            reasoning: true
        }
    }
};

// ===== DOM ELEMENTS =====
const elements = {
    // Theme and status
    themeToggle: document.getElementById('themeToggle'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    
    // Model selection
    modelCards: document.querySelectorAll('.model-card'),
    selectedModelDisplay: document.getElementById('selectedModel'),
    
    // Prompt input
    originalPrompt: document.getElementById('originalPrompt'),
    charCount: document.getElementById('charCount'),
    rulesPreview: document.getElementById('rulesPreview'),
    rulesList: document.getElementById('rulesList'),
    
    // Settings
    enhancementType: document.getElementById('enhancementType'),
    maxLength: document.getElementById('maxLength'),
    temperature: document.getElementById('temperature'),
    tempValue: document.getElementById('tempValue'),
    topP: document.getElementById('topP'),
    topPValue: document.getElementById('topPValue'),
    evaluateQuality: document.getElementById('evaluateQuality'),
    generateAlternatives: document.getElementById('generateAlternatives'),
    alternativesCount: document.getElementById('alternativesCount'),
    numAlternatives: document.getElementById('numAlternatives'),
    
    // Actions
    enhanceBtn: document.getElementById('enhanceBtn'),
    clearBtn: document.getElementById('clearBtn'),
    
    // Results
    resultsSection: document.getElementById('resultsSection'),
    processingTime: document.getElementById('processingTime'),
    modelUsed: document.getElementById('modelUsed'),
    qualityScore: document.getElementById('qualityScore'),
    originalPromptDisplay: document.getElementById('originalPromptDisplay'),
    enhancedPromptDisplay: document.getElementById('enhancedPromptDisplay'),
    originalQuality: document.getElementById('originalQuality'),
    enhancedQuality: document.getElementById('enhancedQuality'),
    qualityImprovement: document.getElementById('qualityImprovement'),
    generatedContent: document.getElementById('generatedContent'),
    generationMetadata: document.getElementById('generationMetadata'),
    alternatives: document.getElementById('alternatives'),
    
    // Export actions
    copyEnhancedBtn: document.getElementById('copyEnhancedBtn'),
    copyGeneratedBtn: document.getElementById('copyGeneratedBtn'),
    exportBtn: document.getElementById('exportBtn'),
    
    // Loading
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingSteps: document.querySelectorAll('.step')
};

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkSystemStatus();
});

function initializeApp() {
    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    
    // Initialize character counter
    updateCharCount();
    
    // Initialize range value displays
    updateRangeValue('temperature', 'tempValue');
    updateRangeValue('topP', 'topPValue');
    
    // Set initial button state
    updateEnhanceButtonState();
    
    console.log('=€ AI Prompt Enhancement Studio initialized');
}

// ===== EVENT LISTENERS =====
function setupEventListeners() {
    // Theme toggle
    elements.themeToggle?.addEventListener('click', toggleTheme);
    
    // Model selection
    elements.modelCards.forEach(card => {
        card.addEventListener('click', () => selectModel(card.dataset.model));
    });
    
    // Prompt input
    elements.originalPrompt?.addEventListener('input', handlePromptInput);
    elements.originalPrompt?.addEventListener('paste', handlePromptPaste);
    
    // Settings
    elements.temperature?.addEventListener('input', () => updateRangeValue('temperature', 'tempValue'));
    elements.topP?.addEventListener('input', () => updateRangeValue('topP', 'topPValue'));
    elements.generateAlternatives?.addEventListener('change', toggleAlternativesCount);
    
    // Actions
    elements.enhanceBtn?.addEventListener('click', enhancePrompt);
    elements.clearBtn?.addEventListener('click', clearAll);
    
    // Export actions
    elements.copyEnhancedBtn?.addEventListener('click', () => copyToClipboard('enhanced'));
    elements.copyGeneratedBtn?.addEventListener('click', () => copyToClipboard('generated'));
    elements.exportBtn?.addEventListener('click', exportResults);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

// ===== THEME MANAGEMENT =====
function toggleTheme() {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
}

function setTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Update theme toggle icon
    const icon = elements.themeToggle?.querySelector('i');
    if (icon) {
        icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
}

// ===== MODEL SELECTION =====
function selectModel(modelId) {
    if (selectedModel === modelId) return;
    
    // Update selected model
    selectedModel = modelId;
    
    // Update UI
    elements.modelCards.forEach(card => {
        card.classList.toggle('selected', card.dataset.model === modelId);
    });
    
    // Update selected model display
    if (elements.selectedModelDisplay) {
        elements.selectedModelDisplay.textContent = MODEL_RULES[modelId]?.name || 'Unknown';
    }
    
    // Show rules preview
    showRulesPreview(modelId);
    
    // Update enhance button state
    updateEnhanceButtonState();
    
    // Add selection animation
    const selectedCard = document.querySelector(`[data-model="${modelId}"]`);
    if (selectedCard) {
        selectedCard.style.transform = 'translateY(-4px) scale(1.02)';
        setTimeout(() => {
            selectedCard.style.transform = '';
        }, 200);
    }
    
    console.log(`> Selected model: ${MODEL_RULES[modelId]?.name}`);
}

function showRulesPreview(modelId) {
    const rules = MODEL_RULES[modelId]?.rules || [];
    
    if (rules.length === 0) {
        elements.rulesPreview.style.display = 'none';
        return;
    }
    
    // Create rules list HTML
    const rulesHTML = rules.map(rule => `
        <div class="rule-item">
            <i class="fas fa-check-circle"></i>
            <span>${rule}</span>
        </div>
    `).join('');
    
    elements.rulesList.innerHTML = rulesHTML;
    elements.rulesPreview.style.display = 'block';
    
    // Add fade-in animation
    elements.rulesPreview.classList.add('fade-in');
}

// ===== PROMPT HANDLING =====
function handlePromptInput() {
    updateCharCount();
    updateEnhanceButtonState();
    
    // Auto-save to localStorage
    localStorage.setItem('draftPrompt', elements.originalPrompt.value);
}

function handlePromptPaste(event) {
    // Allow paste, then update after a short delay
    setTimeout(() => {
        updateCharCount();
        updateEnhanceButtonState();
    }, 10);
}

function updateCharCount() {
    const count = elements.originalPrompt?.value.length || 0;
    if (elements.charCount) {
        elements.charCount.textContent = count.toLocaleString();
    }
}

// ===== SETTINGS MANAGEMENT =====
function updateRangeValue(rangeId, displayId) {
    const range = document.getElementById(rangeId);
    const display = document.getElementById(displayId);
    if (range && display) {
        display.textContent = range.value;
    }
}

function toggleAlternativesCount() {
    const isChecked = elements.generateAlternatives?.checked;
    elements.alternativesCount.style.display = isChecked ? 'flex' : 'none';
}

// ===== PROMPT ENHANCEMENT =====
async function enhancePrompt() {
    if (isProcessing || !selectedModel || !elements.originalPrompt?.value.trim()) {
        return;
    }
    
    isProcessing = true;
    showLoading(true);
    
    try {
        // Prepare request data
        const requestData = {
            prompt: elements.originalPrompt.value.trim(),
            target_model: selectedModel,
            enhancement_type: elements.enhancementType?.value || 'general',
            max_length: parseInt(elements.maxLength?.value) || 512,
            temperature: parseFloat(elements.temperature?.value) || 0.7,
            top_p: parseFloat(elements.topP?.value) || 0.9,
            evaluate_quality: elements.evaluateQuality?.checked ?? true,
            generate_alternatives: elements.generateAlternatives?.checked ?? false,
            num_alternatives: parseInt(elements.numAlternatives?.value) || 3
        };
        
        // Simulate processing steps
        await simulateProcessingSteps();
        
        // Apply model-specific enhancements
        const enhancedPrompt = applyModelEnhancements(requestData.prompt, selectedModel);
        
        // Make API call (simulated for now)
        const response = await makeEnhancementRequest(requestData, enhancedPrompt);
        
        // Display results
        displayResults(response);
        
    } catch (error) {
        console.error('Enhancement failed:', error);
        showError('Failed to enhance prompt. Please try again.');
    } finally {
        isProcessing = false;
        showLoading(false);
    }
}

function applyModelEnhancements(originalPrompt, modelId) {
    const model = MODEL_RULES[modelId];
    if (!model) return originalPrompt;
    
    const enhancements = model.enhancements;
    let enhancedPrompt = originalPrompt;
    
    // Apply prefix and suffix
    if (enhancements.prefix) {
        enhancedPrompt = enhancements.prefix + enhancedPrompt;
    }
    
    if (enhancements.suffix) {
        enhancedPrompt = enhancedPrompt + enhancements.suffix;
    }
    
    // Apply model-specific formatting
    switch (modelId) {
        case 'claude':
            if (enhancements.xmlTags) {
                enhancedPrompt = `<instructions>\n${enhancedPrompt}\n</instructions>`;
            }
            break;
            
        case 'gemini':
            if (enhancements.constraints) {
                enhancedPrompt += '\n\nPlease stay focused on the specific requirements and provide a structured response.';
            }
            break;
            
        case 'grok':
            if (enhancements.realTimeData) {
                enhancedPrompt = 'Using current information and data, ' + enhancedPrompt.toLowerCase();
            }
            break;
    }
    
    return enhancedPrompt;
}

async function makeEnhancementRequest(requestData, enhancedPrompt) {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Simulate response (replace with actual API call)
    const mockResponse = {
        original_prompt: requestData.prompt,
        enhanced_prompt: enhancedPrompt,
        generated_text: generateMockContent(enhancedPrompt, requestData.enhancement_type),
        original_quality: generateMockMetrics(),
        enhanced_quality: generateMockMetrics(true),
        quality_improvement: generateQualityImprovement(),
        generation_metadata: {
            model: MODEL_RULES[selectedModel].name,
            processing_time: Math.floor(Math.random() * 2000) + 1000,
            temperature: requestData.temperature,
            top_p: requestData.top_p,
            max_length: requestData.max_length,
            tokens_used: Math.floor(Math.random() * 500) + 200
        },
        processing_time: Math.floor(Math.random() * 2000) + 1000,
        alternatives: requestData.generate_alternatives ? generateMockAlternatives(requestData.num_alternatives) : null
    };
    
    return mockResponse;
}

function generateMockContent(prompt, type) {
    const templates = {
        general: 'This is a comprehensive response to your enhanced prompt. The system has analyzed your requirements and provided a detailed, structured answer that addresses all key points mentioned in your request.',
        creative_writing: 'Once upon a digital realm, where artificial intelligence and human creativity danced together in perfect harmony, there existed a prompt enhancement system that could transform simple requests into magnificent literary adventures.',
        technical: '## Technical Implementation\n\n### Overview\nThis technical documentation provides a comprehensive guide to implementing the requested functionality.\n\n### Requirements\n- System compatibility\n- Performance optimization\n- Security considerations',
        analysis: '## Data Analysis Results\n\n### Key Findings\n1. Primary insights from the analysis\n2. Statistical significance\n3. Recommendations for action\n\n### Methodology\nThe analysis was conducted using advanced analytical techniques.',
        coding: '```python\n# Enhanced code solution\ndef enhanced_function(parameters):\n    """\n    Optimized implementation based on enhanced prompt\n    """\n    result = process_enhanced_logic(parameters)\n    return result\n```'
    };
    
    return templates[type] || templates.general;
}

function generateMockMetrics(isEnhanced = false) {
    const base = isEnhanced ? 0.8 : 0.6;
    const variance = 0.15;
    
    return {
        specificity: Math.min(1, base + Math.random() * variance),
        clarity: Math.min(1, base + Math.random() * variance),
        completeness: Math.min(1, base + Math.random() * variance),
        actionability: Math.min(1, base + Math.random() * variance),
        overall: Math.min(1, base + Math.random() * variance)
    };
}

function generateQualityImprovement() {
    return {
        specificity: Math.random() * 0.3 + 0.1,
        clarity: Math.random() * 0.3 + 0.1,
        completeness: Math.random() * 0.3 + 0.1,
        actionability: Math.random() * 0.3 + 0.1,
        overall: Math.random() * 0.3 + 0.1
    };
}

function generateMockAlternatives(count) {
    const alternatives = [];
    for (let i = 0; i < count; i++) {
        alternatives.push({
            prompt: `Alternative enhanced prompt variation ${i + 1}`,
            generated_text: `Alternative generated content ${i + 1} with different approach and perspective.`,
            quality_score: Math.random() * 0.3 + 0.7
        });
    }
    return alternatives;
}

// ===== LOADING ANIMATION =====
async function simulateProcessingSteps() {
    const steps = [
        { id: 'step1', delay: 500 },
        { id: 'step2', delay: 800 },
        { id: 'step3', delay: 600 },
        { id: 'step4', delay: 700 }
    ];
    
    for (const step of steps) {
        const stepElement = document.getElementById(step.id);
        if (stepElement) {
            stepElement.classList.add('active');
        }
        await new Promise(resolve => setTimeout(resolve, step.delay));
    }
}

function showLoading(show) {
    if (elements.loadingOverlay) {
        elements.loadingOverlay.classList.toggle('active', show);
        
        if (!show) {
            // Reset steps
            elements.loadingSteps.forEach(step => {
                step.classList.remove('active');
            });
        }
    }
    
    // Disable enhance button during processing
    if (elements.enhanceBtn) {
        elements.enhanceBtn.disabled = show;
    }
}

// ===== RESULTS DISPLAY =====
function displayResults(response) {
    // Show results section
    elements.resultsSection.style.display = 'block';
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Processing info
    elements.processingTime.textContent = response.processing_time;
    elements.modelUsed.textContent = response.generation_metadata.model;
    elements.qualityScore.textContent = Math.round(response.enhanced_quality.overall * 100);
    
    // Prompt comparison
    elements.originalPromptDisplay.textContent = response.original_prompt;
    elements.enhancedPromptDisplay.textContent = response.enhanced_prompt;
    
    // Quality metrics
    displayQualityMetrics(elements.originalQuality, response.original_quality);
    displayQualityMetrics(elements.enhancedQuality, response.enhanced_quality);
    displayQualityImprovement(response.quality_improvement);
    
    // Generated content
    elements.generatedContent.innerHTML = formatGeneratedContent(response.generated_text);
    displayGenerationMetadata(response.generation_metadata);
    
    // Alternatives
    if (response.alternatives) {
        displayAlternatives(response.alternatives);
    }
    
    // Add fade-in animation
    elements.resultsSection.classList.add('fade-in');
}

function displayQualityMetrics(container, metrics) {
    if (!container || !metrics) return;
    
    const metricsHTML = Object.entries(metrics).map(([key, value]) => {
        const percentage = Math.round(value * 100);
        const colorClass = percentage >= 80 ? 'success' : percentage >= 60 ? 'warning' : 'error';
        return `<span class="metric ${colorClass}">${key}: ${percentage}%</span>`;
    }).join('');
    
    container.innerHTML = metricsHTML;
}

function displayQualityImprovement(improvement) {
    if (!elements.qualityImprovement || !improvement) return;
    
    const improvementHTML = `
        <div class="improvement-summary">
            <h4><i class="fas fa-chart-line"></i> Quality Improvement</h4>
            <div class="improvement-grid">
                ${Object.entries(improvement).map(([key, value]) => `
                    <div class="improvement-item">
                        <span class="improvement-label">${key}</span>
                        <span class="improvement-value">+${Math.round(value * 100)}%</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    elements.qualityImprovement.innerHTML = improvementHTML;
}

function formatGeneratedContent(content) {
    // Simple formatting for different content types
    if (content.includes('```')) {
        return content.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
    }
    
    if (content.includes('##')) {
        return content.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    }
    
    return content.replace(/\n/g, '<br>');
}

function displayGenerationMetadata(metadata) {
    if (!elements.generationMetadata || !metadata) return;
    
    const metadataHTML = `
        <div class="metadata-grid">
            <div class="metadata-item">
                <span class="metadata-label">Tokens Used:</span>
                <span class="metadata-value">${metadata.tokens_used}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Temperature:</span>
                <span class="metadata-value">${metadata.temperature}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Top-p:</span>
                <span class="metadata-value">${metadata.top_p}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Max Length:</span>
                <span class="metadata-value">${metadata.max_length}</span>
            </div>
        </div>
    `;
    
    elements.generationMetadata.innerHTML = metadataHTML;
}

function displayAlternatives(alternatives) {
    if (!elements.alternatives || !alternatives.length) return;
    
    const alternativesHTML = `
        <div class="alternatives-container">
            <h4><i class="fas fa-list-alt"></i> Alternative Variations</h4>
            <div class="alternatives-grid">
                ${alternatives.map((alt, index) => `
                    <div class="alternative-item">
                        <h5>Alternative ${index + 1} (Quality: ${Math.round(alt.quality_score * 100)}%)</h5>
                        <div class="alternative-prompt">${alt.prompt}</div>
                        <div class="alternative-content">${alt.generated_text}</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    elements.alternatives.innerHTML = alternativesHTML;
    elements.alternatives.style.display = 'block';
}

// ===== UTILITY FUNCTIONS =====
function updateEnhanceButtonState() {
    const hasPrompt = elements.originalPrompt?.value.trim().length > 0;
    const hasModel = selectedModel !== null;
    
    if (elements.enhanceBtn) {
        elements.enhanceBtn.disabled = !hasPrompt || !hasModel || isProcessing;
    }
}

function clearAll() {
    // Clear prompt input
    if (elements.originalPrompt) {
        elements.originalPrompt.value = '';
    }
    
    // Hide results
    if (elements.resultsSection) {
        elements.resultsSection.style.display = 'none';
    }
    
    // Hide rules preview
    if (elements.rulesPreview) {
        elements.rulesPreview.style.display = 'none';
    }
    
    // Reset model selection
    selectedModel = null;
    elements.modelCards.forEach(card => card.classList.remove('selected'));
    
    if (elements.selectedModelDisplay) {
        elements.selectedModelDisplay.textContent = 'None';
    }
    
    // Update counters
    updateCharCount();
    updateEnhanceButtonState();
    
    // Clear localStorage
    localStorage.removeItem('draftPrompt');
    
    console.log('>ù Cleared all inputs and results');
}

async function copyToClipboard(type) {
    let text = '';
    
    switch (type) {
        case 'enhanced':
            text = elements.enhancedPromptDisplay?.textContent || '';
            break;
        case 'generated':
            text = elements.generatedContent?.textContent || '';
            break;
    }
    
    if (!text) return;
    
    try {
        await navigator.clipboard.writeText(text);
        showToast(`${type.charAt(0).toUpperCase() + type.slice(1)} content copied to clipboard!`);
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        showToast('Failed to copy to clipboard', 'error');
    }
}

function exportResults() {
    const results = {
        timestamp: new Date().toISOString(),
        model: selectedModel,
        original_prompt: elements.originalPromptDisplay?.textContent || '',
        enhanced_prompt: elements.enhancedPromptDisplay?.textContent || '',
        generated_content: elements.generatedContent?.textContent || '',
        processing_time: elements.processingTime?.textContent || '',
        quality_score: elements.qualityScore?.textContent || ''
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `prompt-enhancement-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('Results exported successfully!');
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Remove after delay
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => document.body.removeChild(toast), 300);
    }, 3000);
}

function showError(message) {
    showToast(message, 'error');
}

// ===== KEYBOARD SHORTCUTS =====
function handleKeyboardShortcuts(event) {
    // Ctrl/Cmd + Enter to enhance
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        if (!elements.enhanceBtn?.disabled) {
            enhancePrompt();
        }
    }
    
    // Escape to clear
    if (event.key === 'Escape') {
        clearAll();
    }
    
    // Ctrl/Cmd + / for help (future feature)
    if ((event.ctrlKey || event.metaKey) && event.key === '/') {
        event.preventDefault();
        // Show help modal (to be implemented)
    }
}

// ===== SYSTEM STATUS =====
async function checkSystemStatus() {
    try {
        // Simulate system check
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Update status
        if (elements.statusDot && elements.statusText) {
            elements.statusDot.style.background = 'var(--success-gradient)';
            elements.statusText.textContent = 'System Ready';
        }
        
        console.log(' System status: Ready');
    } catch (error) {
        console.error('L System status check failed:', error);
        
        if (elements.statusDot && elements.statusText) {
            elements.statusDot.style.background = 'var(--warning-gradient)';
            elements.statusText.textContent = 'System Error';
        }
    }
}

// ===== CSS INJECTION FOR DYNAMIC STYLES =====
function injectDynamicStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            transform: translateX(100%);
            transition: transform 0.3s ease-out;
            z-index: 10000;
        }
        
        .toast.show {
            transform: translateX(0);
        }
        
        .toast-success {
            background: var(--success-gradient);
        }
        
        .toast-error {
            background: var(--secondary-gradient);
        }
        
        .improvement-summary {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
        }
        
        .improvement-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: var(--spacing-md);
            margin-top: var(--spacing-md);
        }
        
        .improvement-item {
            text-align: center;
            padding: var(--spacing-sm);
        }
        
        .improvement-label {
            display: block;
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: var(--spacing-xs);
        }
        
        .improvement-value {
            display: block;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            background: var(--success-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: var(--spacing-md);
            margin-top: var(--spacing-md);
        }
        
        .metadata-item {
            text-align: center;
        }
        
        .metadata-label {
            display: block;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: var(--spacing-xs);
        }
        
        .metadata-value {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .alternatives-container {
            margin-top: var(--spacing-xl);
        }
        
        .alternatives-grid {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-lg);
            margin-top: var(--spacing-md);
        }
        
        .alternative-item {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
        }
        
        .alternative-prompt {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: var(--spacing-md);
            font-family: var(--font-mono);
            font-size: 0.875rem;
            margin: var(--spacing-md) 0;
        }
        
        .alternative-content {
            font-size: 0.875rem;
            line-height: 1.6;
            color: var(--text-secondary);
        }
        
        .metric.success { background: rgba(34, 197, 94, 0.1); color: rgb(34, 197, 94); }
        .metric.warning { background: rgba(245, 158, 11, 0.1); color: rgb(245, 158, 11); }
        .metric.error { background: rgba(239, 68, 68, 0.1); color: rgb(239, 68, 68); }
    `;
    
    document.head.appendChild(style);
}

// Initialize dynamic styles
injectDynamicStyles();

// ===== EXPORT FOR TESTING =====
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MODEL_RULES,
        applyModelEnhancements,
        selectModel,
        updateCharCount
    };
}

console.log('<¨ AI Prompt Enhancement Studio - JavaScript loaded successfully!');