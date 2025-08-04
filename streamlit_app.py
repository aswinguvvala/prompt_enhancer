"""
AI Prompt Enhancement Studio - Professional Streamlit Application
Professional-grade prompt optimization with sophisticated UI design
"""

import streamlit as st
import streamlit.components.v1 as components
import asyncio
import time
import os
import sys
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Import backend modules with explicit error handling
try:
    from config import settings
    from simplified_guides import SIMPLIFIED_MODEL_GUIDES
    from models.prompt_enhancer import PromptEnhancer
    import logging
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all backend modules are properly installed.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Streamlit page config
st.set_page_config(
    page_title="AI Prompt Enhancement Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Professional CSS with animations and improved visual design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* CSS Variables for Enhanced Design */
    :root {
        /* Enhanced Dark Theme Colors */
        --bg-primary: #0a0e1a;
        --bg-secondary: #151b2c;
        --bg-tertiary: #1f2937;
        --bg-card: #1e293b;
        --bg-card-hover: #2a3441;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --bg-gradient-hero: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 50%, rgba(64, 255, 218, 0.1) 100%);
        
        /* Enhanced Text Colors */
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --text-accent: #64ffda;
        --text-accent-alt: #a78bfa;
        
        /* Enhanced Accent Colors */
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-pink: #ec4899;
        --accent-orange: #f59e0b;
        --accent-green: #10b981;
        --accent-cyan: #06b6d4;
        
        /* Model Colors */
        --openai-color: #ff8c42;
        --claude-color: #8b5cf6;
        --gemini-color: #10b981;
        --grok-color: #3b82f6;
        
        /* Enhanced Borders & Shadows */
        --border-color: #334155;
        --border-accent: #3b82f6;
        --border-hover: #64748b;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 8px 25px rgba(0, 0, 0, 0.6);
        --shadow-glow: 0 0 25px rgba(59, 130, 246, 0.4);
        --shadow-colored: 0 8px 32px rgba(102, 126, 234, 0.3);
        
        /* Animation Variables */
        --transition-fast: 0.15s ease;
        --transition-normal: 0.25s ease;
        --transition-slow: 0.4s ease;
        --transition-bounce: 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        
        /* Glass Effect */
        --glass-bg: rgba(30, 41, 59, 0.8);
        --glass-border: rgba(148, 163, 184, 0.2);
        --backdrop-blur: blur(16px);
    }
    
    /* Global Styles */
    .stApp {
        background: var(--bg-primary);
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(64, 255, 218, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
        scroll-behavior: smooth;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Enhanced Header styles */
    .professional-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 2rem;
        background: var(--glass-bg);
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        margin-bottom: 2rem;
        backdrop-filter: var(--backdrop-blur);
        -webkit-backdrop-filter: var(--backdrop-blur);
        transition: all var(--transition-normal);
        box-shadow: var(--shadow-sm);
    }
    
    .professional-header:hover {
        background: rgba(30, 41, 59, 0.9);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all var(--transition-normal);
    }
    
    .logo-section:hover {
        transform: translateY(-1px);
    }
    
    .logo-icon {
        font-size: 2rem;
        filter: drop-shadow(0 0 12px rgba(59, 130, 246, 0.6));
        transition: all var(--transition-normal);
        animation: logoGlow 3s ease-in-out infinite alternate;
    }
    
    .logo-section:hover .logo-icon {
        filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.8));
        transform: scale(1.05);
    }
    
    @keyframes logoGlow {
        0% { filter: drop-shadow(0 0 12px rgba(59, 130, 246, 0.6)); }
        100% { filter: drop-shadow(0 0 18px rgba(59, 130, 246, 0.8)); }
    }
    
    .logo-text {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #64ffda 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: logoTextShift 4s ease-in-out infinite;
        letter-spacing: -0.01em;
    }
    
    @keyframes logoTextShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--glass-bg);
        border-radius: 50px;
        border: 1px solid var(--glass-border);
        backdrop-filter: var(--backdrop-blur);
        -webkit-backdrop-filter: var(--backdrop-blur);
        transition: all var(--transition-normal);
        box-shadow: var(--shadow-sm);
    }
    
    .status-indicator:hover {
        background: rgba(30, 41, 59, 0.9);
        border-color: var(--border-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-orange);
        animation: pulse 2s infinite;
    }
    
    .status-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Enhanced Hero section */
    .hero-section {
        text-align: center;
        margin-bottom: 3rem;
        padding: 3rem 0 4rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: var(--bg-gradient-hero);
        border-radius: 50%;
        z-index: -2;
        animation: heroFloat 20s ease-in-out infinite;
    }
    
    .hero-section::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(139, 92, 246, 0.12) 0%, transparent 50%);
        z-index: -1;
        animation: heroGlow 15s ease-in-out infinite alternate;
    }
    
    @keyframes heroFloat {
        0%, 100% { transform: translate(-50%, -50%) rotate(0deg); }
        33% { transform: translate(-48%, -52%) rotate(1deg); }
        66% { transform: translate(-52%, -48%) rotate(-1deg); }
    }
    
    @keyframes heroGlow {
        0% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 900;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 35%, #64ffda 70%, #a78bfa 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(59, 130, 246, 0.4);
        animation: gradientShift 8s ease-in-out infinite;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        font-size: clamp(1.25rem, 3vw, 1.75rem);
        font-weight: 600;
        color: var(--text-accent);
        margin-bottom: 1.5rem;
        text-shadow: 0 0 20px rgba(100, 255, 218, 0.3);
        animation: fadeInUp 0.8s ease-out 0.3s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-description {
        font-size: clamp(1rem, 2vw, 1.2rem);
        color: var(--text-secondary);
        max-width: 900px;
        margin: 0 auto;
        line-height: 1.8;
        animation: fadeInUp 0.8s ease-out 0.6s both;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* Enhanced Model Selection */
    .model-selection-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-icon {
        font-size: 1.3rem;
    }
    
    .model-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    
    /* Enhanced Model Cards */
    .model-card {
        background: var(--bg-card);
        border: 2px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        transform: translateY(0);
        backdrop-filter: blur(10px);
    }
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, transparent 0%, rgba(79, 156, 249, 0.05) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .model-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-4px);
        box-shadow: var(--shadow-glow);
    }
    
    .model-card:active {
        transform: translateY(-2px);
        transition: all 0.1s ease;
    }
    
    .model-card:hover::before {
        opacity: 1;
    }
    
    .model-card.selected {
        border-color: var(--accent-blue);
        background: var(--bg-tertiary);
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
        animation: modelSelect 0.3s ease;
    }
    
    @keyframes modelSelect {
        0% {
            transform: translateY(0) scale(1);
            box-shadow: var(--shadow-md);
        }
        50% {
            transform: translateY(-6px) scale(1.02);
            box-shadow: var(--shadow-glow);
        }
        100% {
            transform: translateY(-2px) scale(1);
            box-shadow: var(--shadow-glow);
        }
    }
    
    /* Enhanced Model Card Elements */
    .model-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        display: block;
        font-weight: 900;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Display', sans-serif;
        letter-spacing: -0.02em;
        transition: all var(--transition-normal);
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .model-card:hover .model-icon {
        transform: scale(1.1);
        text-shadow: 0 4px 16px currentColor;
        animation: iconFloat 2s ease-in-out infinite;
    }
    
    @keyframes iconFloat {
        0%, 100% { transform: scale(1.1) translateY(0px); }
        50% { transform: scale(1.1) translateY(-4px); }
    }
    
    /* Official Company Colors */
    .openai-icon { 
        color: #10A37F; 
        background: linear-gradient(135deg, #10A37F, #0F8B6C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .claude-icon { 
        color: #CC785C; 
        background: linear-gradient(135deg, #CC785C, #B8694A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .gemini-icon { 
        color: #4285F4; 
        background: linear-gradient(135deg, #4285F4, #306FE0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .grok-icon { 
        color: #1DA1F2; 
        background: linear-gradient(135deg, #1DA1F2, #1991DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .model-name {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        color: var(--text-primary);
        transition: color var(--transition-normal);
    }
    
    .model-card:hover .model-name {
        color: var(--text-accent);
    }
    
    .model-description {
        color: var(--text-secondary);
        margin-bottom: 1rem;
        line-height: 1.5;
        font-weight: 400;
    }
    
    .model-features {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .feature-tag {
        background: rgba(79, 156, 249, 0.2);
        color: var(--text-accent);
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(79, 156, 249, 0.3);
        transition: all var(--transition-normal);
    }
    
    .model-card:hover .feature-tag {
        background: rgba(79, 156, 249, 0.3);
        border-color: rgba(79, 156, 249, 0.5);
        transform: translateY(-1px);
    }
    
    /* Enhanced Prompt Section */
    .prompt-section {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        transition: all var(--transition-normal);
    }
    
    .prompt-section:hover {
        border-color: var(--border-hover);
        background: var(--bg-card-hover);
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Enhanced Textarea Styling */
    .stTextArea > div > div > textarea {
        background: rgba(15, 20, 25, 0.8) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        transition: all var(--transition-normal) !important;
        resize: vertical !important;
        min-height: 120px !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        background: rgba(15, 20, 25, 0.95) !important;
    }
    
    .stTextArea > div > div > textarea:hover {
        border-color: var(--border-hover) !important;
    }
    
    /* Enhanced Button Styling */
    .stButton > button {
        background: var(--bg-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all var(--transition-bounce) !important;
        box-shadow: var(--shadow-md) !important;
        position: relative !important;
        overflow: hidden !important;
        z-index: 1 !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, transparent 100%) !important;
        transition: left 0.5s ease !important;
        z-index: -1 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: var(--shadow-glow) !important;
        background: linear-gradient(135deg, #7c93f1 0%, #8b63b8 100%) !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.01) !important;
        transition: all 0.1s ease !important;
    }
    
    /* Loading State Animation */
    .stButton > button[disabled] {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%) !important;
        cursor: not-allowed !important;
        animation: loadingPulse 1.5s ease-in-out infinite !important;
    }
    
    @keyframes loadingPulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    /* Enhanced Results Section */
    .results-section {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        transition: all var(--transition-normal);
    }
    
    .results-section:hover {
        border-color: var(--border-hover);
        background: var(--bg-card-hover);
    }
    
    .results-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        flex-wrap: wrap;
        gap: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .processing-info {
        display: flex;
        gap: 2rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
        align-items: center;
    }
    
    .processing-time, .model-used {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all var(--transition-normal);
    }
    
    .processing-time:hover, .model-used:hover {
        background: rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-1px);
    }
    
    .results-container {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 2rem;
        margin-bottom: 2rem;
        align-items: start;
    }
    
    .comparison-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem 1rem;
        font-size: 2rem;
        color: var(--accent-blue);
        animation: arrowPulse 2s ease-in-out infinite;
    }
    
    @keyframes arrowPulse {
        0%, 100% { transform: translateX(0); opacity: 0.7; }
        50% { transform: translateX(5px); opacity: 1; }
    }
    
    .result-panel {
        background: rgba(15, 20, 25, 0.8);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        overflow: hidden;
        transition: all var(--transition-normal);
        position: relative;
    }
    
    .result-panel:hover {
        border-color: var(--border-hover);
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem;
        background: var(--bg-tertiary);
        border-bottom: 1px solid var(--border-color);
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .panel-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .panel-stats {
        display: flex;
        align-items: center;
        gap: 1rem;
        font-size: 0.85rem;
        color: var(--text-muted);
    }
    
    .word-count {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.3rem 0.6rem;
        background: rgba(100, 255, 218, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(100, 255, 218, 0.2);
    }
    
    .copy-button {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
        padding: 0.4rem 0.8rem !important;
        color: var(--accent-blue) !important;
        font-size: 0.8rem !important;
        transition: all var(--transition-normal) !important;
        cursor: pointer !important;
    }
    
    .copy-button:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    .panel-content {
        padding: 1.5rem;
        max-height: 400px;
        overflow-y: auto;
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .original-panel {
        border-left: 3px solid var(--accent-orange);
    }
    
    .enhanced-panel {
        border-left: 3px solid var(--accent-green);
    }
    
    .enhancement-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1rem 0;
    }
    
    .arrow-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
    }
    
    .arrow-text {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text-accent);
        letter-spacing: 1px;
    }
    
    .arrow {
        font-size: 2rem;
        color: var(--accent-blue);
        animation: pulse-arrow 2s infinite;
    }
    
    @keyframes pulse-arrow {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    /* Action buttons */
    .action-buttons {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .action-btn {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(45, 55, 72, 0.5);
        border-radius: 12px;
        background: rgba(15, 20, 25, 0.8);
        color: #ffffff;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    
    .action-btn:hover {
        background: rgba(36, 43, 61, 0.8);
        border-color: #4f9cf9;
        transform: translateY(-1px);
    }
    
    /* Geometric background elements */
    .geometric-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        overflow: hidden;
    }
    
    .geometric-shape {
        position: absolute;
        border-radius: 50%;
        opacity: 0.1;
    }
    
    .shape-1 {
        width: 200px;
        height: 200px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        top: 10%;
        right: 10%;
        animation: float 6s ease-in-out infinite;
    }
    
    .shape-2 {
        width: 150px;
        height: 150px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        bottom: 20%;
        left: 15%;
        animation: float 8s ease-in-out infinite reverse;
    }
    
    .shape-3 {
        width: 300px;
        height: 300px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        top: 60%;
        right: 30%;
        animation: float 10s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Enhanced Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 20, 25, 0.8);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
        transition: background var(--transition-normal);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }
    
    /* Theme Toggle (Basic Foundation) */
    .theme-toggle {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 50px;
        padding: 0.5rem;
        cursor: pointer;
        transition: all var(--transition-normal);
        backdrop-filter: var(--backdrop-blur);
        z-index: 100;
    }
    
    .theme-toggle:hover {
        background: rgba(30, 41, 59, 0.9);
        transform: scale(1.05);
    }
    
    /* Enhanced Mobile Responsive Design */
    @media (max-width: 1024px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .professional-header {
            padding: 1rem 1.5rem;
        }
        
        .processing-info {
            gap: 1rem;
        }
    }
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.25rem;
        }
        
        .model-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .model-card {
            padding: 1.25rem;
        }
        
        .results-container {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .comparison-arrow {
            transform: rotate(90deg);
            padding: 1rem;
        }
        
        .professional-header {
            flex-direction: column;
            gap: 1rem;
            text-align: center;
        }
        
        .processing-info {
            flex-direction: column;
            gap: 0.5rem;
            align-items: center;
        }
        
        .action-buttons {
            flex-direction: column;
            align-items: center;
        }
        
        .prompt-section, .results-section {
            padding: 1.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        
        .hero-title {
            font-size: 2rem;
        }
        
        .professional-header {
            padding: 1rem;
        }
        
        .model-card {
            padding: 1rem;
        }
        
        .prompt-section, .results-section {
            padding: 1rem;
        }
        
        .panel-header {
            padding: 0.75rem 1rem;
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
        
        .panel-stats {
            gap: 0.5rem;
        }
    }
    
          /* Touch-friendly interactions */
      @media (hover: none) and (pointer: coarse) {
          .model-card:hover {
              transform: none;
          }
          
          .model-card:active {
              transform: scale(0.98);
          }
          
          .stButton > button:hover {
              transform: none;
          }
          
          .stButton > button:active {
              transform: scale(0.98);
          }
      }
      
      /* Light Theme Variables */
      [data-theme="light"] {
          --bg-primary: #f8fafc;
          --bg-secondary: #f1f5f9;
          --bg-tertiary: #e2e8f0;
          --bg-card: #ffffff;
          --bg-card-hover: #f8fafc;
          
          --text-primary: #1e293b;
          --text-secondary: #475569;
          --text-muted: #64748b;
          --text-accent: #0f766e;
          --text-accent-alt: #7c3aed;
          
          --border-color: #e2e8f0;
          --border-hover: #cbd5e1;
          
          --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
          --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.15);
          --shadow-lg: 0 8px 25px rgba(0, 0, 0, 0.2);
          --shadow-xl: 0 12px 40px rgba(0, 0, 0, 0.25);
          --shadow-glow: 0 0 25px rgba(59, 130, 246, 0.3);
          --shadow-glow-intense: 0 0 40px rgba(59, 130, 246, 0.4);
          --shadow-colored: 0 8px 32px rgba(102, 126, 234, 0.2);
          
          --glass-bg: rgba(255, 255, 255, 0.8);
          --glass-border: rgba(148, 163, 184, 0.3);
          
          --accent-blue: #3b82f6;
          --accent-purple: #8b5cf6;
          --accent-pink: #ec4899;
          --accent-green: #10b981;
          --accent-orange: #f59e0b;
      }
      
      /* Light theme body background */
      [data-theme="light"] .main {
          background-image: 
              radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.05) 0%, transparent 50%),
              radial-gradient(circle at 80% 20%, rgba(64, 255, 218, 0.04) 0%, transparent 50%),
              radial-gradient(circle at 40% 40%, rgba(139, 92, 246, 0.03) 0%, transparent 50%);
      }
      
      /* Dynamic theme application */
      .stApp {
          transition: all var(--transition-normal);
      }
      
      /* Theme-aware scrollbar */
      [data-theme="light"] ::-webkit-scrollbar-track {
          background: #f1f5f9;
      }
      
      [data-theme="light"] ::-webkit-scrollbar-thumb {
          background: #cbd5e1;
      }
      
            [data-theme="light"] ::-webkit-scrollbar-thumb:hover {
          background: #3b82f6;
      }
      
      /* Accessibility improvements */
      @media (prefers-reduced-motion: reduce) {
          * {
              animation-duration: 0.01s !important;
              animation-iteration-count: 1 !important;
              transition-duration: 0.01s !important;
          }
      }
      
      /* High contrast support */
      @media (prefers-contrast: high) {
          :root, [data-theme="light"] {
              --border-color: #000000;
              --text-secondary: #000000;
              --text-muted: #000000;
          }
      }
      
      /* Focus indicators for keyboard navigation */
      .stButton > button:focus-visible,
      .model-card:focus-visible {
          outline: 3px solid var(--accent-blue);
          outline-offset: 2px;
          border-radius: var(--radius-md);
      }
      
      /* Enhanced mobile improvements */
      @media (max-width: 640px) {
          .stButton > button {
              padding: 0.75rem 1.5rem;
              font-size: 1rem;
              min-height: 44px; /* Touch target minimum */
          }
          
          .model-card {
              min-height: 120px;
              padding: 1.5rem 1rem;
          }
          
          /* Ensure adequate spacing for touch */
          .model-grid {
              gap: 1.25rem;
          }
          
          /* Better text area on mobile */
          .stTextArea > div > div > textarea {
              font-size: 16px; /* Prevents zoom on iOS */
              line-height: 1.5;
              padding: 1rem;
          }
      }
      
      /* Print styles */
      @media print {
          .geometric-bg,
          .theme-toggle,
          .stButton {
              display: none !important;
          }
          
          .result-panel {
              break-inside: avoid;
              page-break-inside: avoid;
          }
      }
      
      /* Enhanced Settings Panel */
      .settings-panel {
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          padding: 1.5rem;
          backdrop-filter: var(--backdrop-blur);
          margin-bottom: 1rem;
      }
      
      .settings-header {
          color: var(--text-accent);
          font-size: 1.1rem;
          font-weight: 600;
          margin-bottom: 1rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
      }
      
      .settings-icon {
          font-size: 1.3rem;
          filter: drop-shadow(0 0 8px rgba(100, 255, 218, 0.3));
      }
      
      .enhancement-info {
          margin-top: 1.5rem;
          padding: 1rem;
          background: rgba(79, 156, 249, 0.1);
          border-radius: var(--radius-md);
          border-left: 3px solid var(--accent-blue);
      }
      
      .enhancement-badge {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          background: var(--accent-blue);
          color: white;
          padding: 0.25rem 0.75rem;
          border-radius: var(--radius-full);
          font-size: 0.85rem;
          font-weight: 600;
          margin-bottom: 0.5rem;
      }
      
      .enhancement-desc {
          color: var(--text-secondary);
          font-size: 0.85rem;
          margin: 0.5rem 0 0 0;
          line-height: 1.4;
      }
      
      /* Advanced Model Card Animations */
      .model-card {
          position: relative;
          overflow: hidden;
          transition: all var(--transition-normal);
          animation: cardFloat 6s ease-in-out infinite;
      }
      
      .model-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
          transition: left 0.8s;
      }
      
      .model-card:hover::before {
          left: 100%;
      }
      
      .model-card:hover {
          transform: translateY(-8px) scale(1.02);
          box-shadow: var(--shadow-colored);
      }
      
      .model-card.selected {
          animation: selectedPulse 2s ease-in-out infinite;
          box-shadow: 0 0 30px rgba(59, 130, 246, 0.5), var(--shadow-colored);
      }
      
      @keyframes cardFloat {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-2px); }
      }
      
      @keyframes selectedPulse {
          0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.3), var(--shadow-colored); }
          50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.6), var(--shadow-colored); }
      }
      
      /* Model Icon Animations */
      .model-icon {
          position: relative;
          transition: all var(--transition-bounce);
      }
      
      .model-card:hover .model-icon {
          transform: rotateY(360deg) scale(1.1);
          animation: iconGlow 0.8s ease-out;
      }
      
      @keyframes iconGlow {
          0% { box-shadow: 0 0 5px currentColor; }
          50% { box-shadow: 0 0 20px currentColor, 0 0 30px rgba(255, 255, 255, 0.5); }
          100% { box-shadow: 0 0 5px currentColor; }
      }
      
      /* Feature Tag Animations */
      .feature-tag {
          transition: all var(--transition-normal);
          animation: tagFloat 4s ease-in-out infinite;
      }
      
      .feature-tag:nth-child(1) { animation-delay: 0s; }
      .feature-tag:nth-child(2) { animation-delay: 0.5s; }
      
      .model-card:hover .feature-tag {
          transform: translateY(-2px);
          box-shadow: var(--shadow-sm);
      }
      
      @keyframes tagFloat {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-1px); }
      }
      
      /* Button Advanced Interactions */
      .stButton > button {
          position: relative;
          overflow: hidden;
          transition: all var(--transition-normal);
          animation: buttonIdle 8s ease-in-out infinite;
      }
      
      .stButton > button::before {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          width: 0;
          height: 0;
          background: rgba(255, 255, 255, 0.2);
          border-radius: 50%;
          transform: translate(-50%, -50%);
          transition: width 0.6s, height 0.6s;
      }
      
      .stButton > button:hover::before {
          width: 300px;
          height: 300px;
      }
      
      .stButton > button:hover {
          transform: translateY(-3px) scale(1.02);
          box-shadow: var(--shadow-lg);
      }
      
      @keyframes buttonIdle {
          0%, 100% { box-shadow: var(--shadow-sm); }
          50% { box-shadow: var(--shadow-md); }
      }
      
      /* Textarea Advanced Styling */
      .stTextArea > div > div > textarea {
          transition: all var(--transition-normal);
          animation: textareaBreath 10s ease-in-out infinite;
      }
      
      .stTextArea > div > div > textarea:focus {
          transform: scale(1.01);
          box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
          animation: textareaFocus 2s ease-out;
      }
      
      @keyframes textareaBreath {
          0%, 100% { box-shadow: var(--shadow-sm); }
          50% { box-shadow: var(--shadow-md); }
      }
      
      @keyframes textareaFocus {
          0% { box-shadow: 0 0 0 rgba(59, 130, 246, 0); }
          50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.5); }
          100% { box-shadow: 0 0 30px rgba(59, 130, 246, 0.3); }
      }
      
      /* Loading and Processing Animations */
      .processing-indicator {
          animation: processing 1.5s ease-in-out infinite;
      }
      
      @keyframes processing {
          0%, 100% { opacity: 0.6; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.02); }
      }
      
      /* Enhanced Mobile Touch Feedback */
      @media (hover: none) and (pointer: coarse) {
          .model-card:active {
              transform: scale(0.95);
              transition: transform 0.1s;
          }
          
          .stButton > button:active {
              transform: scale(0.95);
              transition: transform 0.1s;
          }
          
          .feature-tag:active {
              transform: scale(0.95);
              transition: transform 0.1s;
          }
      }
      
      /* Accessibility Enhancements */
      @media (prefers-reduced-motion: reduce) {
          * {
              animation-duration: 0.01ms !important;
              animation-iteration-count: 1 !important;
              transition-duration: 0.01ms !important;
          }
      }
      
      /* High Contrast Mode Support */
      @media (prefers-contrast: high) {
          .model-card {
              border: 2px solid;
          }
          
          .feature-tag {
              border: 1px solid;
          }
      }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_enhancer():
    """Initialize the prompt enhancer with caching."""
    try:
        enhancer = PromptEnhancer()
        return enhancer
    except Exception as e:
        st.error(f"Failed to initialize AI model: {str(e)}")
        return None

def check_openai_config():
    """Check if OpenAI API is properly configured."""
    # Try multiple sources for API key
    api_key = None
    
    # First try environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Then try Streamlit secrets (for deployment)
    if not api_key:
        try:
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
        except Exception as e:
            logger.warning(f"Could not access Streamlit secrets: {e}")
    
    # Finally try settings
    if not api_key:
        api_key = settings.OPENAI_API_KEY
    
    if not api_key:
        st.error("""
        🚨 **OpenAI API Key Required**
        
        Please set your OpenAI API key in one of these ways:
        1. **For local development:** Set environment variable `OPENAI_API_KEY=your_key_here`
        2. **For Streamlit Cloud deployment:** Add it to your app's secrets:
           ```toml
           [secrets]
           OPENAI_API_KEY = "your_key_here"
           ```
        3. Add it to a `.env` file in the project root
        
        You can get your API key from: https://platform.openai.com/api-keys
        """)
        return False
    
    logger.info(f"OpenAI API key found: {api_key[:8]}...{api_key[-4:]}")
    return True

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
    
    # Create simplified, effective context prompt for GPT-4o mini
    context_prompt = f"""You are an expert prompt engineer. Your job is to improve user prompts to work better with {model_name}.

TARGET MODEL: {model_name}
DESCRIPTION: {model_description}

KEY RULES FOR {model_name}:
{rules_section}

WHAT TO AVOID:
{avoid_section}

USER'S ORIGINAL PROMPT:
"{original_prompt}"

Please rewrite this prompt to follow the {model_name} best practices above. Make it clear, specific, and effective for {model_name}. Write only the improved prompt - no explanations or additional text.

ENHANCED PROMPT:"""
    
    return context_prompt

async def enhance_prompt_async(enhancer, original_prompt, target_model, enhancement_type):
    """Async wrapper for prompt enhancement."""
    try:
        # Create context injection prompt using the ported system
        context_prompt = create_context_injection_prompt(
            original_prompt, target_model, enhancement_type
        )
        
        # Enhance using the prompt enhancer
        result = await enhancer.enhance_prompt(
            context_injection_prompt=context_prompt,
            original_prompt=original_prompt,
            target_model=target_model,
            enhancement_type=enhancement_type
        )
        
        return result.enhanced_prompt, result.processing_time, result.backend_used
        
    except Exception as e:
        logger.error(f"Enhancement failed: {str(e)}")
        raise e

def run_async_enhancement(enhancer, original_prompt, target_model, enhancement_type):
    """Run async enhancement in sync context."""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            enhance_prompt_async(enhancer, original_prompt, target_model, enhancement_type)
        )
        
        loop.close()
        return result
        
    except Exception as e:
        logger.error(f"Async enhancement failed: {str(e)}")
        raise e

def main():
    """Main Streamlit application with professional UI design."""
    
    # Initialize theme state first
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    
    # Add theme to main container
    st.markdown(f"""
    <script>
        document.documentElement.setAttribute('data-theme', '{st.session_state.theme}');
    </script>
    """, unsafe_allow_html=True)
    
    # Add geometric background with theme
    st.markdown(f"""
    <div class="geometric-bg" data-theme="{st.session_state.theme}">
        <div class="geometric-shape shape-1"></div>
        <div class="geometric-shape shape-2"></div>
        <div class="geometric-shape shape-3"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme toggle functionality
    theme_col1, theme_col2 = st.columns([8, 1])
    with theme_col2:
        if st.button("Theme" if st.session_state.theme == 'dark' else "Theme", 
                    help="Toggle theme", 
                    key="theme_toggle",
                    use_container_width=True):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
    
    # Professional Header
    st.markdown(f"""
    <div class="professional-header" data-theme="{st.session_state.theme}">
        <div class="logo-section">
            <div class="logo-icon">⚡</div>
            <div class="logo-text">PromptCraft AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check OpenAI configuration
    if not check_openai_config():
        st.stop()
    
    # Initialize enhancer
    enhancer = initialize_enhancer()
    if not enhancer:
        st.stop()
    
    # Hero Section
    st.markdown(f"""
    <div class="hero-section" data-theme="{st.session_state.theme}">
        <h1 class="hero-title">PromptCraft AI</h1>
        <h2 class="hero-subtitle">Transform Ideas into Intelligent Prompts</h2>
        <p class="hero-description">
            Supercharge your AI interactions with intelligent prompt optimization. Built for creators, developers, and innovators who demand excellence from their AI conversations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selection Section
    st.markdown('<h3 class="model-selection-title">Choose Your AI Model</h3>', unsafe_allow_html=True)
    
    # Initialize session state for model selection
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'claude'
    
    # Create model cards in a grid
    col1, col2, col3, col4 = st.columns(4)
    
    model_configs = {
        'openai': {
            'name': 'OpenAI GPT-4',
            'description': 'Advanced reasoning with structured thinking',
            'features': ['Step-by-step reasoning', 'Detailed explanations'],
            'icon_class': 'openai-icon',
            'icon_text': 'GPT'
        },
        'claude': {
            'name': 'Anthropic Claude',
            'description': 'Thoughtful and nuanced responses',
            'features': ['Deep analysis', 'Ethical reasoning'],
            'icon_class': 'claude-icon',
            'icon_text': 'Claude'
        },
        'gemini': {
            'name': 'Google Gemini',
            'description': 'Systematic and comprehensive analysis',
            'features': ['Structured approach', 'Clear explanations'],
            'icon_class': 'gemini-icon',
            'icon_text': 'Gemini'
        },
        'grok': {
            'name': 'xAI Grok',
            'description': 'Practical and direct responses',
            'features': ['Real-time insights', 'Actionable advice'],
            'icon_class': 'grok-icon',
            'icon_text': 'Grok'
        }
    }
    
    columns = [col1, col2, col3, col4]
    model_keys = list(model_configs.keys())
    
    for i, (model_key, config) in enumerate(model_configs.items()):
        with columns[i]:
            selected_class = "selected" if st.session_state.selected_model == model_key else ""
            
            card_html = f"""
            <div class="model-card {selected_class}" onclick="selectModel('{model_key}')">
                <div class="model-icon {config['icon_class']}">{config['icon_text']}</div>
                <h4 class="model-name">{config['name']}</h4>
            </div>
            """
            
            if st.button(f"Select {config['name']}", key=f"btn_{model_key}", use_container_width=True):
                st.session_state.selected_model = model_key
                st.rerun()
    
    # Display selected model cards with selection styling
    for i, (model_key, config) in enumerate(model_configs.items()):
        with columns[i]:
            selected_class = "selected" if st.session_state.selected_model == model_key else ""
            
            card_html = f"""
            <div class="model-card {selected_class}">
                <div class="model-icon {config['icon_class']}">{config['icon_text']}</div>
                <h4 class="model-name">{config['name']}</h4>
                <p class="model-description">{config['description']}</p>
                <div class="model-features">
                    <span class="feature-tag">{config['features'][0]}</span>
                    <span class="feature-tag">{config['features'][1]}</span>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    
    # Prompt Input Section
    st.markdown(f"""
    <div class="prompt-section" data-theme="{st.session_state.theme}">
        <h3 class="section-title">
            <span>💭</span>
            What's on your mind?
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two-column layout for input and settings
    input_col, settings_col = st.columns([2, 1])
    
    with input_col:
        original_prompt = st.text_area(
            "",
            height=200,
            placeholder="How to use this app: 1) Select your target AI model above 2) Enter your original prompt here 3) Click 'Enhance' to optimize it with model-specific techniques 4) Copy the enhanced result for better AI responses",
            label_visibility="collapsed",
            key="prompt_input"
        )
        
        # Character counter
        char_count = len(original_prompt) if original_prompt else 0
        st.markdown(f'<div style="text-align: right; color: #8892b0; font-size: 0.85rem; margin-top: 0.5rem;">{char_count:,} characters</div>', unsafe_allow_html=True)
        
        # Selected model info
        selected_config = model_configs[st.session_state.selected_model]
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(79, 156, 249, 0.1); 
                    border-radius: 12px; font-size: 0.9rem; color: #b8c5d6; 
                    border-left: 3px solid #4f9cf9;">
            <span>Selected: </span>
            <span>{selected_config['name']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with settings_col:
        st.markdown(f"""
        <div class="settings-panel" data-theme="{st.session_state.theme}">
            <h4 class="settings-header">
                <span class="settings-icon">Settings</span>
                Enhancement Settings
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Set standard length (hidden from user)
        max_length = 600
        
        st.markdown("""
        <div class="enhancement-info">
            <span class="enhancement-badge">Professional Enhancement</span>
            <p class="enhancement-desc">Model-specific optimization for better AI responses</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhancement buttons
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Check if prompt is valid for enhancement
        prompt_is_valid = bool(original_prompt and original_prompt.strip())
        
        enhance_button = st.button(
            "Transform My Prompt",
            type="primary",
            disabled=not prompt_is_valid,
            use_container_width=True,
            help="Transform your prompt with AI-powered intelligence" if prompt_is_valid else "Enter some text to enhance your prompt"
        )
        
        clear_button = st.button(
            "Clear All",
            use_container_width=True,
            help="Clear all input and results"
        )
        
        if clear_button:
            st.session_state.clear()
            st.rerun()
    
    # Enhancement Processing and Results
    if enhance_button and original_prompt.strip():
        with st.spinner("🎨 Crafting your perfect prompt..."):
            try:
                start_time = time.time()
                
                # Run enhancement (always use comprehensive)
                enhanced_prompt, processing_time, backend_used = run_async_enhancement(
                    enhancer, original_prompt, st.session_state.selected_model, "comprehensive"
                )
                
                # Store results in session state
                st.session_state.last_original = original_prompt
                st.session_state.last_enhanced = enhanced_prompt
                st.session_state.last_model = st.session_state.selected_model
                st.session_state.last_processing_time = processing_time
                st.session_state.last_backend = backend_used
                
                st.success(f"Prompt transformed in {processing_time:.2f}s using {backend_used}")
                st.rerun()
                
            except Exception as e:
                error_message = str(e)
                
                # Provide specific error messages based on the type of error
                if "Authentication failed" in error_message:
                    st.error("**API Authentication Error**")
                    st.error("Your OpenAI API key appears to be invalid or expired.")
                    st.info("Please check your API key in the deployment settings and ensure it's correct.")
                elif "Access forbidden" in error_message:
                    st.error("**API Access Forbidden**")
                    st.error("Your OpenAI API key doesn't have permission to access the required model.")
                    st.info("Please check your OpenAI account has sufficient credits and permissions.")
                elif "Rate limit exceeded" in error_message:
                    st.error("**Rate Limit Exceeded**")
                    st.error("Too many requests to OpenAI API. Please wait a moment and try again.")
                    st.info("This usually resolves itself in a few minutes.")
                elif "not configured" in error_message:
                    st.error("**Configuration Error**")
                    st.error("OpenAI API key is not properly configured.")
                    st.info("""
                    **For Streamlit Cloud deployment:**
                    1. Go to your app settings
                    2. Click on 'Secrets' 
                    3. Add: `OPENAI_API_KEY = "your-key-here"`
                    """)
                else:
                    st.error(f"**Enhancement failed:** {error_message}")
                    st.info("Please check your internet connection and try again.")
                
                # Log the full error for debugging
                logger.error(f"Enhancement failed: {error_message}")
                
                # Show fallback result if possible
                if original_prompt:
                    st.warning("Showing fallback enhancement...")
                    fallback_enhanced = f"As an expert assistant, I'll help you with the following: {original_prompt}\n\nI'll provide comprehensive information and clear explanations to ensure you get the most helpful response possible."
                    
                    st.session_state.last_original = original_prompt
                    st.session_state.last_enhanced = fallback_enhanced
                    st.session_state.last_model = st.session_state.selected_model
                    st.session_state.last_processing_time = 0
                    st.session_state.last_backend = "fallback"
    
    # Results Display Section
    if hasattr(st.session_state, 'last_enhanced'):
        st.markdown(f"""
        <div class="results-section" data-theme="{st.session_state.theme}">
            <div class="results-header">
                <h3 class="section-title">
                    <span>Results</span>
                    Your Enhanced Prompt
                </h3>
                <div class="processing-info">
                    <span class="processing-time">
                        <span>Time:</span>
                        Generated in {st.session_state.last_processing_time:.2f}s
                    </span>
                    <span class="model-used">
                        <span>Model:</span>
                        Powered by {model_configs[st.session_state.last_model]['name']}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Side-by-side results comparison
        result_col1, arrow_col, result_col2 = st.columns([5, 1, 5])
        
        with result_col1:
            st.markdown(f"""
            <div class="result-panel original-panel" data-theme="{st.session_state.theme}">
                <div class="panel-header">
                    <span>Original</span>
                    <span>Before</span>
                    <div class="panel-stats">
                        <span class="word-count">{len(st.session_state.last_original.split())} words</span>
                        <span class="char-count">{len(st.session_state.last_original)} chars</span>
                    </div>
                </div>
                <div class="panel-content">{st.session_state.last_original}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with arrow_col:
            st.markdown("""
            <div class="enhancement-arrow">
                <div class="arrow-container">
                    <span class="arrow-text">ENHANCED</span>
                    <div class="arrow">→</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            st.markdown(f"""
            <div class="result-panel enhanced-panel" data-theme="{st.session_state.theme}">
                <div class="panel-header">
                    <span>Enhanced</span>
                    <span>After</span>
                    <div class="panel-stats">
                        <span class="word-count">{len(st.session_state.last_enhanced.split())} words</span>
                        <span class="char-count">{len(st.session_state.last_enhanced)} chars</span>
                    </div>
                </div>
                <div class="panel-content">{st.session_state.last_enhanced}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        action_col1, action_col2, action_col3 = st.columns([1, 1, 1])
        
        with action_col2:
            if st.button("Copy Result", use_container_width=True):
                # Use Streamlit's built-in method to copy to clipboard
                st.code(st.session_state.last_enhanced, language=None)
                st.success("Enhanced prompt ready to copy!")

if __name__ == "__main__":
    main()
    /* Enhanced Card Interactions for Streamlit */
    .stColumn > div {
        transition: all var(--transition-normal);
    }
    
    .stColumn:hover > div {
        transform: translateY(-2px);
    }
    
    /* Streamlit Button Enhancements */
    .stButton > button {
        background: var(--glass-bg) \!important;
        border: 1px solid var(--glass-border) \!important;
        border-radius: 12px \!important;
        color: var(--text-primary) \!important;
        font-weight: 500 \!important;
        padding: 0.75rem 1.5rem \!important;
        transition: all var(--transition-normal) \!important;
        backdrop-filter: var(--backdrop-blur) \!important;
        -webkit-backdrop-filter: var(--backdrop-blur) \!important;
        box-shadow: var(--shadow-sm) \!important;
    }
    
    .stButton > button:hover {
        background: rgba(30, 41, 59, 0.9) \!important;
        border-color: var(--accent-blue) \!important;
        box-shadow: var(--shadow-glow) \!important;
        transform: translateY(-1px) \!important;
    }
    
    .stButton > button:active {
        transform: translateY(0) \!important;
        box-shadow: var(--shadow-sm) \!important;
    }
    
    /* Enhanced Textarea */
    .stTextArea > div > div > textarea {
        background: var(--glass-bg) \!important;
        border: 1px solid var(--glass-border) \!important;
        border-radius: 12px \!important;
        color: var(--text-primary) \!important;
        backdrop-filter: var(--backdrop-blur) \!important;
        -webkit-backdrop-filter: var(--backdrop-blur) \!important;
        box-shadow: var(--shadow-sm) \!important;
        transition: all var(--transition-normal) \!important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-blue) \!important;
        box-shadow: var(--shadow-glow) \!important;
        transform: translateY(-1px) \!important;
    }
    
    /* Card Grid Layout Enhancement */
    .row-widget.stHorizontal {
        gap: 1rem;
    }
    
    .row-widget.stHorizontal > div {
        padding: 0.5rem;
    }
    
    /* Floating Animation for Interactive Elements */
    @keyframes streamlitFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-2px) rotate(0.5deg); }
        66% { transform: translateY(2px) rotate(-0.5deg); }
    }
    
    .stColumn:nth-child(even) {
        animation: streamlitFloat 6s ease-in-out infinite;
        animation-delay: 0.5s;
    }
    
    .stColumn:nth-child(odd) {
        animation: streamlitFloat 8s ease-in-out infinite;
        animation-delay: 1s;
    }
    
    /* Enhanced Progress Indicators */
    .stProgress > div > div {
        background: var(--accent-blue) \!important;
        border-radius: 6px \!important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.4) \!important;
    }
    
    /* Success/Error States */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px \!important;
        backdrop-filter: var(--backdrop-blur) \!important;
        -webkit-backdrop-filter: var(--backdrop-blur) \!important;
        border: 1px solid var(--glass-border) \!important;
        box-shadow: var(--shadow-sm) \!important;
    }
EOF < /dev/null