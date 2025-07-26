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
            'Structure: Task context � Tone � Background � Detailed task',
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
            'Four-component structure: Persona � Task � Context � Format',
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
    
    // Settings
    maxLength: document.getElementById('maxLength'),
    
    // Actions
    enhanceBtn: document.getElementById('enhanceBtn'),
    clearBtn: document.getElementById('clearBtn'),
    
    // Results
    resultsSection: document.getElementById('resultsSection'),
    processingTime: document.getElementById('processingTime'),
    modelUsed: document.getElementById('modelUsed'),
    originalPromptDisplay: document.getElementById('originalPromptDisplay'),
    enhancedPromptDisplay: document.getElementById('enhancedPromptDisplay'),
    alternatives: document.getElementById('alternatives'),
    
    // Export actions
    copyEnhancedBtn: document.getElementById('copyEnhancedBtn'),
    exportBtn: document.getElementById('exportBtn'),
    
    // Loading
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingSteps: document.querySelectorAll('.step')
};

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkSystemStatusDetailed();
});

function initializeApp() {
    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    
    // Initialize character counter
    updateCharCount();
    
    // Settings initialized with default values
    
    // Set initial button state
    updateEnhanceButtonState();
    
    console.log('=� AI Prompt Enhancement Studio initialized');
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
    
    // Settings event listeners removed (simplified system)
    
    // Actions
    elements.enhanceBtn?.addEventListener('click', enhancePrompt);
    elements.clearBtn?.addEventListener('click', clearAll);
    
    // Export actions
    elements.copyEnhancedBtn?.addEventListener('click', () => copyToClipboard('enhanced'));
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

function updateProgressMessage(message) {
    // Update progress message for users during long AI processing
    const enhancedSection = document.querySelector('.enhanced-prompt');
    if (enhancedSection) {
        enhancedSection.innerHTML = `<div class="progress-message">
            <div class="spinner"></div>
            <p>${message}</p>
            <small>Processing may take up to 2 minutes for complex prompts...</small>
        </div>`;
    }
}

// ===== SETTINGS MANAGEMENT =====
// Range value update function removed - simplified UI system


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
            enhancement_type: 'comprehensive',
            max_length: parseInt(elements.maxLength?.value) || 512
        };
        
        
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
    try {
        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 second timeout for complex AI processing
        
        // Add progress monitoring
        let progressMessages = [
            { time: 5000, message: "Analyzing your prompt with AI models..." },
            { time: 15000, message: "Applying model-specific optimization rules..." },
            { time: 30000, message: "AI is generating enhanced version..." },
            { time: 60000, message: "Complex prompt detected, using advanced processing..." },
            { time: 90000, message: "Finalizing enhancement with quality checks..." }
        ];
        
        let progressIndex = 0;
        const progressInterval = setInterval(() => {
            if (progressIndex < progressMessages.length) {
                const progress = progressMessages[progressIndex];
                updateProgressMessage(progress.message);
                progressIndex++;
            }
        }, 10000); // Update every 10 seconds
        
        // Make actual API call to backend
        const response = await fetch('http://localhost:8001/enhance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        clearInterval(progressInterval); // Clear progress updates
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        return result;
        
    } catch (error) {
        clearTimeout(timeoutId);
        clearInterval(progressInterval); // Clear progress updates on error
        console.error('Enhancement API call failed:', error);
        console.log('Error type:', error.name);
        console.log('Error message:', error.message);
        console.log('Full error:', error);
        
        // Comprehensive error handling with specific feedback
        let errorMessage = 'Unknown error occurred. Check browser console for details.';
        let generatedMessage = 'Unable to generate content. Check console logs for debugging information.';
        let suggestions = ['Open browser developer tools to see detailed error information'];
        
        if (error.name === 'AbortError') {
            errorMessage = `AI processing timed out after 120 seconds.`;
            generatedMessage = 'The AI enhancement is taking longer than expected. The system has multiple timeout fallbacks:';
            suggestions = [
                'Try again - the system uses faster models on retry',
                'Simplify your prompt to reduce processing complexity', 
                'The backend automatically tries quick enhancement mode after timeouts',
                'Check if your AI model (Ollama) is responding properly',
                'Verify sufficient system resources are available'
            ];
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Cannot connect to backend server. Server may not be running.';
            generatedMessage = 'The AI Prompt Enhancement Studio backend is not accessible.';
            suggestions = [
                'Start the backend server: python start_server.py',
                'Check if port 8001 is available and not blocked',
                'Verify the server is running on localhost:8001',
                'Check firewall settings and network connectivity'
            ];
        } else if (error.message.includes('500')) {
            errorMessage = 'Server internal error. The AI model may be unavailable.';
            generatedMessage = 'The enhancement engine encountered an unexpected issue.';
            suggestions = [
                'Check if Ollama is running: ollama serve',
                'Verify the llama3.2:3b model is available: ollama list',
                'Try restarting the backend server',
                'Check server logs for detailed error information'
            ];
        } else if (error.message.includes('429')) {
            errorMessage = 'Rate limit exceeded. Too many requests sent recently.';
            generatedMessage = 'Please wait a moment before trying again.';
            suggestions = [
                'Wait 30 seconds before making another request',
                'Consider caching frequently used prompts',
                'Use Quick Mode to reduce server load',
                'Check if multiple instances are running'
            ];
        } else if (error.message.includes('413')) {
            errorMessage = 'Prompt too large. The input exceeds size limits.';
            generatedMessage = 'Your prompt may be too long for processing.';
            suggestions = [
                'Reduce prompt length to under 2000 characters',
                'Break complex prompts into smaller parts',
                'Remove unnecessary details from your prompt',
                'Use Ultra Mode for complex prompts if needed'
            ];
        }
        
        // Create detailed error display
        const errorDetails = suggestions.length > 0 ? 
            `\n\nSuggested solutions:\n${suggestions.map(s => `• ${s}`).join('\n')}` : '';
        
        generatedMessage += errorDetails;
        
        // Fallback response for when API fails
        return {
            original_prompt: requestData.prompt,
            enhanced_prompt: errorMessage,
            generated_text: generatedMessage,
            generation_metadata: {
                model: MODEL_RULES[selectedModel]?.name || 'Unknown',
                processing_time: 0,
                temperature: requestData.temperature,
                top_p: requestData.top_p,
                max_length: requestData.max_length,
                tokens_used: 0
            },
            processing_time: 0,
            alternatives: null,
            error: error.message,
            fallback_used: true
        };
    }
}

// Removed mock generation functions - now using real AI API calls

// ===== LOADING ANIMATION =====

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
    
    // Prompt comparison
    elements.originalPromptDisplay.textContent = response.original_prompt;
    elements.enhancedPromptDisplay.textContent = response.enhanced_prompt;
    
    // Alternatives
    if (response.alternatives) {
        displayAlternatives(response.alternatives);
    }
    
    // Add fade-in animation
    elements.resultsSection.classList.add('fade-in');
}

// Quality metrics functions removed - no longer displaying fake metrics


function displayAlternatives(alternatives) {
    if (!elements.alternatives || !alternatives.length) return;
    
    const alternativesHTML = `
        <div class="alternatives-container">
            <h4><i class="fas fa-list-alt"></i> Alternative Variations</h4>
            <div class="alternatives-grid">
                ${alternatives.map((alt, index) => {
                    return `
                        <div class="alternative-item">
                            <h5>Alternative ${index + 1}</h5>
                            <div class="alternative-prompt">${alt.prompt}</div>
                            <div class="alternative-content">${alt.generated_text}</div>
                        </div>
                    `;
                }).join('')}
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
    
    console.log('>� Cleared all inputs and results');
}

async function copyToClipboard(type) {
    let text = '';
    
    if (type === 'enhanced') {
        text = elements.enhancedPromptDisplay?.textContent || '';
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
        processing_time: elements.processingTime?.textContent || ''
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

function showProgressUpdate(message, type = 'info') {
    // Create progress notification
    const progressDiv = document.createElement('div');
    progressDiv.className = `progress-update ${type}`;
    progressDiv.innerHTML = `
        <div class="progress-icon">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-triangle' : 'fa-info-circle'}"></i>
        </div>
        <div class="progress-message">${message}</div>
    `;
    
    // Add to status area or create floating notification
    const statusArea = document.querySelector('.status-indicator') || document.body;
    statusArea.appendChild(progressDiv);
    
    // Auto-remove after delay
    setTimeout(() => {
        if (progressDiv.parentNode) {
            progressDiv.parentNode.removeChild(progressDiv);
        }
    }, 4000);
}

// Enhanced system status checker with progress updates
async function checkSystemStatusDetailed() {
    try {
        showProgressUpdate('Checking system status...', 'info');
        
        const response = await fetch('http://localhost:8001/system/performance');
        const data = await response.json();
        
        if (response.ok) {
            const systemPerf = data.system_performance;
            const cacheHitRate = systemPerf.cache.hit_rate * 100;
            
            showProgressUpdate(`System ready - Cache efficiency: ${cacheHitRate.toFixed(1)}%`, 'success');
            
            // Update status display
            if (elements.statusDot && elements.statusText) {
                elements.statusDot.style.background = 'var(--success-gradient)';
                elements.statusText.textContent = `Ready (${cacheHitRate.toFixed(0)}% cached)`;
            }
        } else {
            throw new Error('Status check failed');
        }
        
    } catch (error) {
        showProgressUpdate('System status check failed - some features may be limited', 'error');
        
        if (elements.statusDot && elements.statusText) {
            elements.statusDot.style.background = 'var(--warning-gradient)';
            elements.statusText.textContent = 'Limited';
        }
    }
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

// ===== DEBUGGING FUNCTIONS =====
// Test API connectivity from browser console
window.testAPI = async function() {
    console.log('🔍 Testing API connectivity...');
    
    try {
        // Test health endpoint
        const healthResponse = await fetch('http://localhost:8001/health');
        if (healthResponse.ok) {
            const healthData = await healthResponse.json();
            console.log('✅ Health check passed:', healthData);
        } else {
            console.error('❌ Health check failed:', healthResponse.status, healthResponse.statusText);
            return false;
        }
        
        // Test enhance endpoint
        const testRequest = {
            prompt: "Test prompt",
            target_model: "claude",
            enhancement_type: "comprehensive",
            max_length: 100,
            evaluate_quality: false
        };
        
        console.log('🔧 Testing enhance endpoint...');
        const enhanceResponse = await fetch('http://localhost:8001/enhance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testRequest)
        });
        
        if (enhanceResponse.ok) {
            const enhanceData = await enhanceResponse.json();
            console.log('✅ Enhance endpoint working:', enhanceData);
            return true;
        } else {
            console.error('❌ Enhance endpoint failed:', enhanceResponse.status, enhanceResponse.statusText);
            return false;
        }
        
    } catch (error) {
        console.error('❌ API test failed:', error);
        return false;
    }
};

console.log('🚀 AI Prompt Enhancement Studio - JavaScript loaded successfully!');
console.log('💡 Use testAPI() in console to debug API connectivity');