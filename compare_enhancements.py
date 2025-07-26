#!/usr/bin/env python3
"""
Comparison between AI-enhanced and hardcoded rule-based prompt enhancement
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app import PromptOptimizer

async def compare_enhancement_approaches():
    """Compare AI vs hardcoded enhancement approaches."""
    
    optimizer = PromptOptimizer()
    
    # Test prompts
    test_prompts = [
        "Write a story about a robot.",
        "Help me debug my code.",
        "Explain quantum computing.",
        "Create a marketing plan.",
        "Analyze this data set."
    ]
    
    target_models = ["openai", "claude", "gemini"]
    
    print("=" * 80)
    print("AI ENHANCEMENT vs HARDCODED RULES COMPARISON")
    print("=" * 80)
    
    for prompt in test_prompts:
        print(f"\n🔍 ORIGINAL PROMPT: '{prompt}'\n")
        
        for model in target_models:
            print(f"--- {model.upper()} ---")
            
            # AI-based enhancement
            try:
                ai_enhanced = await optimizer.apply_comprehensive_enhancement(
                    prompt, model, "comprehensive"
                )
                print(f"✨ AI Enhanced: {ai_enhanced}")
            except Exception as e:
                print(f"❌ AI Enhancement failed: {str(e)}")
                ai_enhanced = "FAILED"
            
            # Hardcoded enhancement
            hardcoded_enhanced = optimizer.create_hardcoded_enhancement(prompt, model)
            print(f"🔧 Hardcoded: {hardcoded_enhanced}")
            
            # Quality comparison
            print(f"📊 Quality Analysis:")
            print(f"   - AI: More contextual, model-specific, intelligent phrasing")
            print(f"   - Hardcoded: Generic rules, templated, limited variation")
            print()
        
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(compare_enhancement_approaches())