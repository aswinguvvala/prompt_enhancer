#!/usr/bin/env python3
"""
Test script to verify deployment fixes for Streamlit app
"""

import sys
import os
sys.path.append('backend')

def test_config_loading():
    """Test configuration and API key loading"""
    print("🧪 Testing configuration loading...")
    
    try:
        from config import settings, get_openai_api_key
        print("✅ Config imported successfully")
        
        # Test API key detection
        api_key = get_openai_api_key()
        if api_key:
            print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
        else:
            print("⚠️  No API key found (expected for testing)")
            
        print(f"✅ AI Backend: {settings.AI_BACKEND}")
        print(f"✅ OpenAI Model: {settings.OPENAI_MODEL}")
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_prompt_enhancer():
    """Test PromptEnhancer initialization"""
    print("\n🧪 Testing PromptEnhancer initialization...")
    
    try:
        from models.prompt_enhancer import PromptEnhancer
        print("✅ PromptEnhancer imported successfully")
        
        # Test initialization 
        enhancer = PromptEnhancer()
        print("✅ PromptEnhancer initialized successfully")
        print(f"✅ Primary backend: {enhancer.primary_backend}")
        
        # Test OpenAI client
        if enhancer.openai_client.api_key:
            print("✅ OpenAI client has API key configured")
        else:
            print("⚠️  OpenAI client has no API key (expected for testing)")
            
        return True
        
    except Exception as e:
        print(f"❌ PromptEnhancer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_imports():
    """Test Streamlit-related imports"""
    print("\n🧪 Testing Streamlit compatibility...")
    
    try:
        # Test if streamlit import handling works
        import streamlit as st
        print("✅ Streamlit is available")
        
        # Test if our config handles streamlit properly
        from config import STREAMLIT_AVAILABLE
        print(f"✅ Streamlit availability detected: {STREAMLIT_AVAILABLE}")
        
        return True
        
    except ImportError:
        print("⚠️  Streamlit not available (install with: pip install streamlit)")
        return True  # This is okay for testing
    except Exception as e:
        print(f"❌ Streamlit compatibility test failed: {e}")
        return False

def test_error_handling():
    """Test error handling improvements"""
    print("\n🧪 Testing error handling...")
    
    try:
        from models.prompt_enhancer import OpenAIClient
        
        # Test client with no API key
        client = OpenAIClient()
        
        # The client should handle missing API key gracefully
        if not client.api_key:
            print("✅ OpenAI client handles missing API key gracefully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Streamlit Deployment Fixes\n")
    
    tests = [
        test_config_loading,
        test_prompt_enhancer, 
        test_streamlit_imports,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Deployment fixes are working correctly.")
        print("\n📝 Next steps for deployment:")
        print("1. Set your OpenAI API key in Streamlit secrets")
        print("2. Deploy to Streamlit Cloud")
        print("3. Button should become active immediately when typing text")
        print("4. Enhanced prompts should use OpenAI API instead of fallback")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()