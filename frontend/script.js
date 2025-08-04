// AI Prompt Enhancement Studio - Interactive JavaScript

class PromptEnhancementStudio {
    constructor() {
        this.selectedModel = 'claude'; // Default to Claude
        this.apiBaseUrl = 'http://localhost:8001'; // FastAPI backend
        this.isEnhancing = false;
        this.currentTheme = localStorage.getItem('theme') || 'dark';
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupModelSelection();
        this.setupCharacterCounter();
        this.updateSelectedModelInfo();
        this.updateEnhanceButtonState(); // Initialize button state
        this.checkBackendHealth();
        this.initializeTheme();
        this.initializeParticleSystem(); // Initialize particle background
    }

    setupEventListeners() {
        // Model selection
        const modelCards = document.querySelectorAll('.model-card');
        modelCards.forEach(card => {
            card.addEventListener('click', () => this.selectModel(card));
            card.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.selectModel(card);
                }
            });
        });

        // Enhancement button
        const enhanceBtn = document.getElementById('enhanceBtn');
        enhanceBtn.addEventListener('click', () => this.enhancePrompt());

        // Clear button
        const clearBtn = document.getElementById('clearBtn');
        clearBtn.addEventListener('click', () => this.clearAll());

        // Copy button
        const copyBtn = document.getElementById('copyBtn');
        copyBtn.addEventListener('click', () => this.copyEnhancedPrompt());

        // Export button
        const exportBtn = document.getElementById('exportBtn');
        exportBtn.addEventListener('click', () => this.exportResults());
        
        // Panel copy buttons
        const panelCopyBtns = document.querySelectorAll('.panel-copy-btn');
        panelCopyBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const target = e.currentTarget.dataset.target;
                this.copyFromPanel(target);
            });
        });
        
        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        themeToggle.addEventListener('click', () => this.toggleTheme());

        // Text area input with enhanced features
        const promptInput = document.getElementById('originalPrompt');
        promptInput.addEventListener('input', (e) => {
            this.updateCharCount();
            this.updateEnhanceButtonState(); // Immediate button state update
            this.validateInputDebounced(e.target.value);
        });
        
        promptInput.addEventListener('keydown', (e) => {
            // Enhanced keyboard shortcuts
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.enhancePrompt();
            } else if (e.ctrlKey && e.key === 'l') {
                e.preventDefault();
                this.clearAll();
            }
        });
        
        // Add paste enhancement
        promptInput.addEventListener('paste', (e) => {
            // Use a small delay to ensure the pasted content is in the DOM
            setTimeout(() => {
                this.updateCharCount();
                this.updateEnhanceButtonState(); // Immediate button state update
                this.validateInputDebounced(promptInput.value); // Use current value from DOM
            }, 10); // Small delay to ensure paste content is processed
        });
        
        // Focus enhancement
        promptInput.addEventListener('focus', () => {
            document.querySelector('.input-section').classList.add('focused');
        });
        
        promptInput.addEventListener('blur', () => {
            document.querySelector('.input-section').classList.remove('focused');
        });

        // Max length setting
        const maxLengthInput = document.getElementById('maxLength');
        maxLengthInput.addEventListener('change', () => this.updateSettings());
    }

    setupModelSelection() {
        // Set Claude as default selected
        const claudeCard = document.querySelector('[data-model="claude"]');
        if (claudeCard) {
            claudeCard.classList.add('selected');
            claudeCard.setAttribute('tabindex', '0');
        }

        // Set tabindex for other cards
        const modelCards = document.querySelectorAll('.model-card');
        modelCards.forEach((card, index) => {
            if (!card.classList.contains('selected')) {
                card.setAttribute('tabindex', '0');
            }
        });
    }

    selectModel(card) {
        // Remove selection from all cards
        document.querySelectorAll('.model-card').forEach(c => {
            c.classList.remove('selected');
        });

        // Add selection to clicked card
        card.classList.add('selected');
        this.selectedModel = card.dataset.model;
        
        this.updateSelectedModelInfo();
        this.updateEnhanceButtonState(); // Update button state immediately
        this.showToast(`Selected ${this.getModelDisplayName(this.selectedModel)}`, 'info');
    }

    getModelDisplayName(modelKey) {
        const modelNames = {
            'openai': 'OpenAI GPT-4',
            'claude': 'Anthropic Claude 4 (Opus/Sonnet)',
            'gemini': 'Google Gemini 1.5 Pro',
            'grok': 'xAI Grok 3/4'
        };
        return modelNames[modelKey] || modelKey;
    }

    updateSelectedModelInfo() {
        const selectedModelName = document.getElementById('selectedModelName');
        selectedModelName.textContent = this.getModelDisplayName(this.selectedModel);
    }

    setupCharacterCounter() {
        this.updateCharCount();
        this.setupAutoResize();
    }

    setupAutoResize() {
        const promptInput = document.getElementById('originalPrompt');
        
        // Auto-resize function
        const autoResize = () => {
            promptInput.style.height = 'auto';
            promptInput.style.height = Math.max(120, promptInput.scrollHeight) + 'px';
        };
        
        // Set up auto-resize on input
        promptInput.addEventListener('input', autoResize);
        promptInput.addEventListener('paste', () => setTimeout(autoResize, 0));
        
        // Initial resize
        autoResize();
    }

    updateCharCount() {
        const promptInput = document.getElementById('originalPrompt');
        const charCount = document.getElementById('charCount');
        const count = promptInput.value.length;
        
        charCount.textContent = count.toLocaleString();
        
        // Add visual feedback for character count
        charCount.classList.remove('warning', 'danger');
        if (count > 8000) {
            charCount.classList.add('danger');
        } else if (count > 5000) {
            charCount.classList.add('warning');
        }
        
        // Update enhance button state
        this.updateEnhanceButtonState();
        
        // Update real-time analytics
        this.updateTextareaAnalytics(promptInput.value);
        
        // Auto-resize textarea
        this.autoResizeTextarea(promptInput);
    }
    
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        const newHeight = Math.max(120, Math.min(textarea.scrollHeight, 400));
        textarea.style.height = newHeight + 'px';
        
        // Add smooth transition
        if (!textarea.style.transition) {
            textarea.style.transition = 'height 0.2s ease';
        }
    }

    updateSettings() {
        // Settings updated - could add validation here
        console.log('Settings updated');
        this.updateEnhanceButtonState();
    }

    updateEnhanceButtonState() {
        const promptInput = document.getElementById('originalPrompt');
        const enhanceBtn = document.getElementById('enhanceBtn');
        const count = promptInput.value.trim().length;
        
        // Button is enabled if:
        // 1. There's text content (more than 0 characters)
        // 2. A model is selected (this.selectedModel exists)
        // 3. Not currently enhancing
        const canEnhance = count > 0 && this.selectedModel && !this.isEnhancing;
        
        enhanceBtn.disabled = !canEnhance;
        
        // Enhanced visual feedback for button state
        if (canEnhance) {
            enhanceBtn.style.opacity = '1';
            enhanceBtn.style.cursor = 'pointer';
            enhanceBtn.style.transform = '';
            enhanceBtn.classList.remove('disabled-state');
        } else {
            enhanceBtn.style.opacity = this.isEnhancing ? '0.9' : '0.4';
            enhanceBtn.style.cursor = 'not-allowed';
            enhanceBtn.style.transform = 'scale(0.98)';
            enhanceBtn.classList.add('disabled-state');
        }
        
        // Update button text based on state
        const btnText = enhanceBtn.childNodes[2];
        if (count === 0) {
            btnText.textContent = 'Enter prompt to enhance';
        } else if (!this.selectedModel) {
            btnText.textContent = 'Select a model first';
        } else if (!this.isEnhancing) {
            btnText.textContent = 'Enhance with AI Intelligence';
        }
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateStatusIndicator('online', 'Connected');
            } else {
                this.updateStatusIndicator('warning', 'Limited');
            }
        } catch (error) {
            console.warn('Backend health check failed:', error);
            this.updateStatusIndicator('offline', 'Offline');
        }
    }

    updateStatusIndicator(status, text) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        // Remove existing status classes
        statusDot.classList.remove('status-online', 'status-warning', 'status-offline');
        
        // Add new status
        statusDot.classList.add(`status-${status}`);
        statusText.textContent = text;
    }

    async enhancePrompt() {
        const promptInput = document.getElementById('originalPrompt');
        const originalPrompt = promptInput.value.trim();

        if (!originalPrompt) {
            this.showToast('Please enter a prompt to enhance', 'error');
            return;
        }

        if (this.isEnhancing) {
            return;
        }

        this.isEnhancing = true;
        this.showLoadingOverlay();
        this.updateEnhanceButton(true);

        const startTime = performance.now();

        try {
            const maxLength = parseInt(document.getElementById('maxLength').value);
            
            const response = await fetch(`${this.apiBaseUrl}/enhance`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    original_prompt: originalPrompt,
                    target_model: this.selectedModel,
                    enhancement_type: 'comprehensive',
                    max_length: maxLength
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const processingTime = performance.now() - startTime;

            this.displayResults({
                original: originalPrompt,
                enhanced: data.enhanced_prompt,
                processingTime: processingTime,
                modelUsed: this.getModelDisplayName(this.selectedModel),
                metadata: data.metadata || {}
            });

            this.showToast('Prompt enhanced successfully!', 'success');

        } catch (error) {
            console.error('Enhancement failed:', error);
            this.showToast('Enhancement failed. Please check your connection and try again.', 'error');
            
            // Show a basic enhancement as fallback
            this.displayFallbackResult(originalPrompt);
        } finally {
            this.isEnhancing = false;
            this.hideLoadingOverlay();
            this.updateEnhanceButton(false);
            this.updateEnhanceButtonState(); // Ensure button state is properly reset
        }
    }

    displayResults(results) {
        // Update the display elements with enhanced animation
        const originalDisplay = document.getElementById('originalPromptDisplay');
        const enhancedDisplay = document.getElementById('enhancedPromptDisplay');
        const processingTimeEl = document.getElementById('processingTime');
        const modelUsedEl = document.getElementById('modelUsed');
        
        // Animate text appearance
        originalDisplay.style.opacity = '0';
        enhancedDisplay.style.opacity = '0';
        
        setTimeout(() => {
            originalDisplay.textContent = results.original;
            enhancedDisplay.textContent = results.enhanced;
            processingTimeEl.textContent = results.processingTime.toFixed(0);
            modelUsedEl.textContent = results.modelUsed;
            
            // Fade in with stagger
            originalDisplay.style.transition = 'opacity 0.5s ease';
            enhancedDisplay.style.transition = 'opacity 0.5s ease 0.2s';
            originalDisplay.style.opacity = '1';
            enhancedDisplay.style.opacity = '1';
        }, 300);

        // Show results section with animation
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';
        resultsSection.style.opacity = '0';
        resultsSection.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            resultsSection.style.transition = 'all 0.6s ease';
            resultsSection.style.opacity = '1';
            resultsSection.style.transform = 'translateY(0)';
        }, 100);
        
        // Smooth scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 500);

        // Store results for export
        this.lastResults = results;
        
        // Update word counts
        this.updateWordCounts(results.original, results.enhanced);
        
        // Add confetti effect for successful enhancement
        this.triggerSuccessAnimation();
    }
    
    triggerSuccessAnimation() {
        // Simple success particle effect
        const particles = [];
        const colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'];
        
        for (let i = 0; i < 15; i++) {
            const particle = document.createElement('div');
            particle.style.cssText = `
                position: fixed;
                width: 8px;
                height: 8px;
                background: ${colors[Math.floor(Math.random() * colors.length)]};
                border-radius: 50%;
                pointer-events: none;
                z-index: 9999;
                left: 50%;
                top: 30%;
                animation: particleExplode 1.5s ease-out forwards;
            `;
            
            particle.style.setProperty('--random-x', `${(Math.random() - 0.5) * 400}px`);
            particle.style.setProperty('--random-y', `${(Math.random() - 0.5) * 300}px`);
            
            document.body.appendChild(particle);
            particles.push(particle);
        }
        
        // Clean up particles
        setTimeout(() => {
            particles.forEach(particle => {
                if (particle.parentNode) {
                    particle.parentNode.removeChild(particle);
                }
            });
        }, 1500);
    }

    displayFallbackResult(originalPrompt) {
        const enhancedPrompt = `As an expert assistant, I would like to help you with the following request: ${originalPrompt}

Please provide detailed information and comprehensive explanations to ensure the response meets your needs effectively. I'll structure my response clearly and include relevant examples where appropriate to maximize the value and clarity of the information provided.`;

        this.displayResults({
            original: originalPrompt,
            enhanced: enhancedPrompt,
            processingTime: 0,
            modelUsed: 'Fallback Enhancement',
            metadata: { fallback: true }
        });
    }

    showLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = 'flex';
        overlay.style.opacity = '0';
        document.body.style.overflow = 'hidden';
        
        // Smooth fade in
        requestAnimationFrame(() => {
            overlay.style.transition = 'opacity 0.3s ease';
            overlay.style.opacity = '1';
        });
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        
        // Smooth fade out
        overlay.style.transition = 'opacity 0.3s ease';
        overlay.style.opacity = '0';
        
        setTimeout(() => {
            overlay.style.display = 'none';
            document.body.style.overflow = 'auto';
        }, 300);
    }

    updateEnhanceButton(isLoading) {
        const enhanceBtn = document.getElementById('enhanceBtn');
        const btnIcon = enhanceBtn.querySelector('.btn-icon');
        const btnText = enhanceBtn.childNodes[2]; // Text node after icon and space

        if (isLoading) {
            btnIcon.textContent = '⏳';
            btnText.textContent = 'Enhancing...';
            enhanceBtn.classList.add('loading');
        } else {
            btnIcon.textContent = 'Processing...';
            btnText.textContent = 'Enhance with AI Intelligence';
            enhanceBtn.classList.remove('loading');
        }
    }

    async copyEnhancedPrompt() {
        const enhancedText = document.getElementById('enhancedPromptDisplay').textContent;
        
        if (!enhancedText) {
            this.showToast('No enhanced prompt to copy', 'error');
            return;
        }

        try {
            await navigator.clipboard.writeText(enhancedText);
            this.showToast('Enhanced prompt copied to clipboard!', 'success');
        } catch (error) {
            console.error('Copy failed:', error);
            
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = enhancedText;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            this.showToast('Enhanced prompt copied to clipboard!', 'success');
        }
    }

    exportResults() {
        if (!this.lastResults) {
            this.showToast('No results to export', 'error');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            selectedModel: this.selectedModel,
            originalPrompt: this.lastResults.original,
            enhancedPrompt: this.lastResults.enhanced,
            processingTime: this.lastResults.processingTime,
            modelUsed: this.lastResults.modelUsed,
            metadata: this.lastResults.metadata
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `prompt-enhancement-${Date.now()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        this.showToast('Results exported successfully!', 'success');
    }

    clearAll() {
        // Clear input
        document.getElementById('originalPrompt').value = '';
        
        // Hide results
        document.getElementById('resultsSection').style.display = 'none';
        
        // Reset character count
        this.updateCharCount();
        
        // Clear stored results
        this.lastResults = null;
        
        this.showToast('All content cleared', 'info');
    }

    showToast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toastContainer');
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        // Create toast content with icon
        const icons = {
            success: '✅',
            error: '❌',
            warning: 'Warning',
            info: 'Info'
        };
        
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-message">${message}</span>
            <button class="toast-close" aria-label="Close notification">×</button>
        `;
        
        // Add close button functionality
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => this.removeToast(toast));
        
        container.appendChild(toast);
        
        // Auto-remove toast after duration
        const timeoutId = setTimeout(() => {
            this.removeToast(toast);
        }, duration);
        
        // Store timeout ID for potential cancellation
        toast.timeoutId = timeoutId;
        
        return toast;
    }
    
    removeToast(toast) {
        if (toast.timeoutId) {
            clearTimeout(toast.timeoutId);
        }
        
        if (toast.parentNode) {
            toast.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }
    }

    // Utility methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    formatNumber(num) {
        return num.toLocaleString();
    }

    sanitizeHTML(str) {
        const temp = document.createElement('div');
        temp.textContent = str;
        return temp.innerHTML;
    }
    
    // Enhanced input validation
    validateInput(text) {
        const warnings = [];
        const errors = [];
        
        if (text.length > 10000) {
            errors.push('Prompt is too long (max 10,000 characters)');
        } else if (text.length > 8000) {
            warnings.push('Very long prompt may take longer to process');
        }
        
        if (text.trim().length < 10) {
            warnings.push('Prompt seems quite short - consider adding more detail');
        }
        
        return { warnings, errors, isValid: errors.length === 0 };
    }
    
    // Debounced input validation
    validateInputDebounced = this.debounce((text) => {
        const validation = this.validateInput(text);
        
        // Show validation feedback
        const inputContainer = document.querySelector('.input-section');
        let validationEl = inputContainer.querySelector('.validation-feedback');
        
        if (!validationEl) {
            validationEl = document.createElement('div');
            validationEl.className = 'validation-feedback';
            inputContainer.appendChild(validationEl);
        }
        
        if (validation.errors.length > 0) {
            validationEl.innerHTML = `<div class="validation-error">Error: ${validation.errors[0]}</div>`;
            validationEl.style.display = 'block';
        } else if (validation.warnings.length > 0) {
            validationEl.innerHTML = `<div class="validation-warning">Warning: ${validation.warnings[0]}</div>`;
            validationEl.style.display = 'block';
        } else {
            validationEl.style.display = 'none';
        }
    }, 500);
    
    // Copy from panel functionality
    async copyFromPanel(targetId) {
        const targetElement = document.getElementById(targetId);
        if (!targetElement) {
            this.showToast('Nothing to copy', 'error');
            return;
        }
        
        const text = targetElement.textContent;
        if (!text.trim()) {
            this.showToast('No content to copy', 'error');
            return;
        }
        
        try {
            await navigator.clipboard.writeText(text);
            this.showToast('Copied to clipboard!', 'success', 2000);
            
            // Visual feedback on the button
            const btn = document.querySelector(`[data-target="${targetId}"]`);
            const originalIcon = btn.querySelector('.btn-icon').textContent;
            btn.querySelector('.btn-icon').textContent = '✓';
            btn.style.color = 'var(--accent-green)';
            
            setTimeout(() => {
                btn.querySelector('.btn-icon').textContent = originalIcon;
                btn.style.color = '';
            }, 1500);
            
        } catch (error) {
            console.error('Copy failed:', error);
            this.showToast('Failed to copy', 'error');
        }
    }
    
    // Update word counts
    updateWordCounts(originalText, enhancedText) {
        const countWords = (text) => {
            return text.trim().split(/\s+/).filter(word => word.length > 0).length;
        };
        
        const originalCount = countWords(originalText);
        const enhancedCount = countWords(enhancedText);
        
        document.getElementById('originalWordCount').textContent = `${originalCount.toLocaleString()} words`;
        document.getElementById('enhancedWordCount').textContent = `${enhancedCount.toLocaleString()} words`;
        
        // Add improvement indicator
        const improvement = enhancedCount - originalCount;
        const enhancedCountEl = document.getElementById('enhancedWordCount');
        
        if (improvement > 0) {
            enhancedCountEl.style.color = 'var(--accent-green)';
            enhancedCountEl.title = `+${improvement} words added`;
        } else if (improvement < 0) {
            enhancedCountEl.style.color = 'var(--accent-orange)';
            enhancedCountEl.title = `${Math.abs(improvement)} words removed`;
        } else {
            enhancedCountEl.style.color = 'var(--text-muted)';
            enhancedCountEl.title = 'Same word count';
        }
    }
    
    // Theme functionality
    initializeTheme() {
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        this.updateThemeIcon();
    }
    
    toggleTheme() {
        this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        localStorage.setItem('theme', this.currentTheme);
        this.updateThemeIcon();
        this.showToast(`Switched to ${this.currentTheme} theme`, 'info', 2000);
    }
    
    updateThemeIcon() {
        const themeIcon = document.querySelector('.theme-icon');
        themeIcon.textContent = this.currentTheme === 'dark' ? 'Dark' : 'Light';
    }

    // Initialize particle system
    initializeParticleSystem() {
        const canvas = document.getElementById('particleCanvas');
        if (canvas) {
            this.particleSystem = new ParticleSystem(canvas);
            this.particleSystem.init();
        }
    }
    
    // Real-time textarea analytics
    updateTextareaAnalytics(text) {
        const analyticsPanel = document.getElementById('textareaAnalytics');
        const inputSection = document.querySelector('.input-section');
        
        if (text.length > 10) {
            analyticsPanel.style.display = 'grid';
            inputSection.classList.add('analytics-active');
            
            const analytics = this.calculateTextAnalytics(text);
            
            document.getElementById('wordCount').textContent = analytics.wordCount;
            document.getElementById('sentenceCount').textContent = analytics.sentenceCount;
            document.getElementById('readingTime').textContent = analytics.readingTime;
            document.getElementById('complexityScore').textContent = analytics.complexity.label;
            document.getElementById('clarityScore').textContent = analytics.clarity.label;
            document.getElementById('specificityScore').textContent = analytics.specificity.label;
            
            this.updateComplexityIndicator(analytics.complexity.score);
            document.getElementById('complexityItem').className = `analytics-item ${analytics.complexity.class}`;
            document.getElementById('clarityItem').className = `analytics-item ${analytics.clarity.class}`;
            document.getElementById('specificityItem').className = `analytics-item ${analytics.specificity.class}`;
        } else {
            analyticsPanel.style.display = 'none';
            inputSection.classList.remove('analytics-active');
        }
    }
    
    calculateTextAnalytics(text) {
        const words = text.trim().split(/\s+/).filter(word => word.length > 0);
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        const readingTimeMinutes = Math.ceil(words.length / 200);
        const readingTime = readingTimeMinutes < 1 ? '<1m' : `${readingTimeMinutes}m`;
        
        const avgWordsPerSentence = sentences.length > 0 ? words.length / sentences.length : 0;
        const avgWordLength = words.length > 0 ? words.join('').length / words.length : 0;
        let complexityScore = Math.min(5, Math.round((avgWordsPerSentence / 12) + (avgWordLength / 4)));
        
        const complexity = this.getComplexityLevel(complexityScore);
        const clarity = this.getClarityLevel(5 - complexityScore + (sentences.length > 0 ? 1 : 0));
        const specificity = this.getSpecificityLevel(this.calculateSpecificity(text));
        
        return {
            wordCount: words.length,
            sentenceCount: sentences.length,
            readingTime,
            complexity,
            clarity,
            specificity
        };
    }
    
    getComplexityLevel(score) {
        if (score <= 1) return { label: 'Basic', class: 'excellent', score: 1 };
        if (score <= 2) return { label: 'Simple', class: 'good', score: 2 };
        if (score <= 3) return { label: 'Medium', class: 'good', score: 3 };
        if (score <= 4) return { label: 'Complex', class: 'warning', score: 4 };
        return { label: 'Advanced', class: 'danger', score: 5 };
    }
    
    getClarityLevel(score) {
        if (score >= 4) return { label: 'Excellent', class: 'excellent' };
        if (score >= 3) return { label: 'Clear', class: 'good' };
        if (score >= 2) return { label: 'Decent', class: 'warning' };
        return { label: 'Unclear', class: 'danger' };
    }
    
    calculateSpecificity(text) {
        let score = 3;
        if (/\d+/.test(text)) score += 0.5;
        if (/(create|build|implement|analyze|optimize|design)/.test(text.toLowerCase())) score += 0.5;
        if (/(some|many|various|things|stuff|good|nice)/.test(text.toLowerCase())) score -= 0.5;
        return Math.max(1, Math.min(5, score));
    }
    
    getSpecificityLevel(score) {
        if (score >= 4) return { label: 'Precise', class: 'excellent' };
        if (score >= 3) return { label: 'Good', class: 'good' };
        if (score >= 2) return { label: 'Vague', class: 'warning' };
        return { label: 'Too Vague', class: 'danger' };
    }
    
    updateComplexityIndicator(score) {
        for (let i = 1; i <= 5; i++) {
            const dot = document.getElementById(`dot${i}`);
            if (i <= score) {
                dot.classList.add('active');
            } else {
                dot.classList.remove('active');
            }
        }
    }
}

// Particle System Class
class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.mouse = { x: 0, y: 0 };
        this.maxParticles = 80;
        this.connectionDistance = 120;
        this.mouseRadius = 150;
        
        // Bind methods
        this.animate = this.animate.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleResize = this.handleResize.bind(this);
    }

    init() {
        this.setupCanvas();
        this.createParticles();
        this.setupEventListeners();
        this.animate();
    }

    setupCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        
        // Set canvas style for better rendering
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
    }

    setupEventListeners() {
        window.addEventListener('mousemove', this.handleMouseMove);
        window.addEventListener('resize', this.handleResize);
    }

    handleMouseMove(e) {
        this.mouse.x = e.clientX;
        this.mouse.y = e.clientY;
    }

    handleResize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    createParticles() {
        this.particles = [];
        for (let i = 0; i < this.maxParticles; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1,
                opacity: Math.random() * 0.5 + 0.3,
                color: this.getRandomColor()
            });
        }
    }

    getRandomColor() {
        const colors = [
            'rgba(102, 126, 234, 0.8)',  // Blue
            'rgba(139, 92, 246, 0.8)',   // Purple
            'rgba(100, 255, 218, 0.8)',  // Cyan
            'rgba(59, 130, 246, 0.8)',   // Light Blue
            'rgba(167, 139, 250, 0.8)'   // Light Purple
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    updateParticles() {
        this.particles.forEach(particle => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Bounce off walls
            if (particle.x < 0 || particle.x > this.canvas.width) {
                particle.vx *= -1;
            }
            if (particle.y < 0 || particle.y > this.canvas.height) {
                particle.vy *= -1;
            }

            // Keep particles in bounds
            particle.x = Math.max(0, Math.min(this.canvas.width, particle.x));
            particle.y = Math.max(0, Math.min(this.canvas.height, particle.y));

            // Mouse interaction
            const dx = this.mouse.x - particle.x;
            const dy = this.mouse.y - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < this.mouseRadius) {
                const force = (this.mouseRadius - distance) / this.mouseRadius;
                const angle = Math.atan2(dy, dx);
                particle.vx -= Math.cos(angle) * force * 0.01;
                particle.vy -= Math.sin(angle) * force * 0.01;
            }

            // Add slight random movement
            particle.vx += (Math.random() - 0.5) * 0.01;
            particle.vy += (Math.random() - 0.5) * 0.01;

            // Limit velocity
            const maxVel = 2;
            particle.vx = Math.max(-maxVel, Math.min(maxVel, particle.vx));
            particle.vy = Math.max(-maxVel, Math.min(maxVel, particle.vy));

            // Add drag
            particle.vx *= 0.99;
            particle.vy *= 0.99;
        });
    }

    drawParticles() {
        this.particles.forEach(particle => {
            this.ctx.save();
            this.ctx.globalAlpha = particle.opacity;
            this.ctx.fillStyle = particle.color;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.restore();
        });
    }

    drawConnections() {
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const dx = this.particles[i].x - this.particles[j].x;
                const dy = this.particles[i].y - this.particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.connectionDistance) {
                    const opacity = (this.connectionDistance - distance) / this.connectionDistance * 0.3;
                    
                    this.ctx.save();
                    this.ctx.globalAlpha = opacity;
                    this.ctx.strokeStyle = 'rgba(100, 255, 218, 0.4)';
                    this.ctx.lineWidth = 1;
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
                    this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
                    this.ctx.stroke();
                    this.ctx.restore();
                }
            }
        }
    }

    animate() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Update and draw
        this.updateParticles();
        this.drawConnections();
        this.drawParticles();

        // Continue animation
        requestAnimationFrame(this.animate);
    }

    destroy() {
        window.removeEventListener('mousemove', this.handleMouseMove);
        window.removeEventListener('resize', this.handleResize);
    }
}

// CSS animations for toasts
const toastAnimations = `
@keyframes slideOutRight {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}

@keyframes particleExplode {
    0% {
        transform: translate(0, 0) scale(1);
        opacity: 1;
    }
    100% {
        transform: translate(var(--random-x), var(--random-y)) scale(0);
        opacity: 0;
    }
}

.status-online {
    background: var(--accent-green) !important;
}

.status-warning {
    background: var(--accent-orange) !important;
}

.status-offline {
    background: var(--accent-pink) !important;
}
`;

// Add animations to document
const styleSheet = document.createElement('style');
styleSheet.textContent = toastAnimations;
document.head.appendChild(styleSheet);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PromptEnhancementStudio();
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to enhance
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const enhanceBtn = document.getElementById('enhanceBtn');
        if (!enhanceBtn.disabled) {
            enhanceBtn.click();
        }
    }
    
    // Escape to clear/cancel
    if (e.key === 'Escape') {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay.style.display === 'flex') {
            // Don't close loading overlay with escape - let the process complete
            return;
        }
        
        // Clear if no enhancement in progress
        document.getElementById('clearBtn').click();
    }
});

// Add service worker for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').then((registration) => {
            console.log('SW registered: ', registration);
        }).catch((registrationError) => {
            console.log('SW registration failed: ', registrationError);
        });
    });
}

// Performance monitoring
window.addEventListener('load', () => {
    if (window.performance) {
        const loadTime = window.performance.timing.loadEventEnd - window.performance.timing.navigationStart;
        console.log(`Page load time: ${loadTime}ms`);
    }
});

// Handle network status
window.addEventListener('online', () => {
    console.log('Network connection restored');
    // Could trigger a health check here
});

window.addEventListener('offline', () => {
    console.log('Network connection lost');
    // Update UI to reflect offline status
});

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PromptEnhancementStudio;
}