#!/usr/bin/env python3
"""
Deployment Verification Script for AI Prompt Enhancement Studio
Tests all core functionality before deployment
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_imports():
    """Test that all required imports work."""
    print("🔍 Testing imports...")
    try:
        import streamlit
        from backend.config import settings
        from backend.simplified_guides import SIMPLIFIED_MODEL_GUIDES
        from backend.models.prompt_enhancer import PromptEnhancer
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test OpenAI configuration."""
    print("\n🔧 Testing configuration...")
    try:
        from backend.config import settings
        
        # Check OpenAI API key
        if not settings.OPENAI_API_KEY:
            print("❌ OpenAI API key not configured")
            return False
        
        if settings.OPENAI_API_KEY.startswith("sk-"):
            print("✅ OpenAI API key format is correct")
        else:
            print("⚠️  OpenAI API key format may be incorrect")
        
        # Check backend configuration
        if settings.AI_BACKEND == "openai" and settings.FALLBACK_BACKEND == "openai":
            print("✅ Backend configured for OpenAI-only")
        else:
            print("⚠️  Backend configuration may need adjustment")
        
        print(f"✅ Using model: {settings.OPENAI_MODEL}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_model_guides():
    """Test model guides are loaded."""
    print("\n📚 Testing model guides...")
    try:
        from backend.simplified_guides import SIMPLIFIED_MODEL_GUIDES
        
        models = list(SIMPLIFIED_MODEL_GUIDES.keys())
        print(f"✅ Loaded {len(models)} model guides: {', '.join(models)}")
        
        for model_id, guide in SIMPLIFIED_MODEL_GUIDES.items():
            if 'rules' in guide and len(guide['rules']) > 0:
                print(f"✅ {guide['name']}: {len(guide['rules'])} optimization rules")
            else:
                print(f"⚠️  {guide['name']}: No rules found")
        
        return True
        
    except Exception as e:
        print(f"❌ Model guides test failed: {e}")
        return False

async def test_prompt_enhancer():
    """Test prompt enhancer initialization."""
    print("\n🤖 Testing AI integration...")
    try:
        from backend.models.prompt_enhancer import PromptEnhancer
        
        enhancer = PromptEnhancer()
        print("✅ PromptEnhancer initialized successfully")
        
        # Test health check
        health_status = await enhancer.health_check()
        
        openai_status = health_status.get('backends', {}).get('openai', {})
        if openai_status.get('available'):
            print("✅ OpenAI API connection successful")
        else:
            print("⚠️  OpenAI API connection may have issues")
            
        if openai_status.get('api_configured'):
            print("✅ OpenAI API key is configured")
        else:
            print("❌ OpenAI API key not properly configured")
        
        return openai_status.get('available', False) and openai_status.get('api_configured', False)
        
    except Exception as e:
        print(f"❌ AI integration test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 AI Prompt Enhancement Studio - Deployment Verification\n")
    
    tests = [
        ("Imports", test_imports()),
        ("Configuration", test_configuration()),
        ("Model Guides", test_model_guides()),
    ]
    
    # Run async test
    print("\n🤖 Testing AI integration...")
    ai_test_result = asyncio.run(test_prompt_enhancer())
    tests.append(("AI Integration", ai_test_result))
    
    # Summary
    print("\n" + "="*50)
    print("📊 VERIFICATION SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Ready for deployment!")
        print("\nTo run the Streamlit app:")
        print("streamlit run streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())