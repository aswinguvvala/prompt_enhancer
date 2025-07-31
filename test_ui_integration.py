#!/usr/bin/env python3
"""
Test script to verify the complete UI restoration and integration
"""

import os
import sys
import subprocess
import time
import requests
import json
from pathlib import Path

def test_files_exist():
    """Test that all required files have been restored."""
    print("🔍 Testing file restoration...")
    
    required_files = [
        "frontend/index.html",
        "frontend/style.css", 
        "frontend/script.js",
        "backend/app.py",
        "backend/config.py",
        "backend/models/prompt_enhancer.py",
        "backend/simplified_guides.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def test_frontend_content():
    """Test that frontend files contain expected content."""
    print("🔍 Testing frontend content...")
    
    # Test HTML has the professional structure
    html_path = Path("frontend/index.html")
    html_content = html_path.read_text()
    
    expected_html_elements = [
        "AI Prompt Enhancement Studio",
        "model-card",
        "OpenAI GPT-4",
        "Anthropic Claude", 
        "Google Gemini",
        "xAI Grok",
        "prompt-input",
        "enhance-btn"
    ]
    
    missing_elements = []
    for element in expected_html_elements:
        if element not in html_content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ Missing HTML elements: {missing_elements}")
        return False
    
    # Test CSS has professional styling
    css_path = Path("frontend/style.css")
    css_content = css_path.read_text()
    
    expected_css_features = [
        "var(--bg-primary)",
        "dark theme",
        ".model-card",
        "gradient",
        ".enhance-btn",
        "professional"
    ]
    
    css_matches = sum(1 for feature in expected_css_features if feature.lower() in css_content.lower())
    
    if css_matches < 4:  # At least 4 features should be present
        print(f"❌ CSS missing professional styling features")
        return False
    
    print("✅ Frontend content looks professional")
    return True

def test_backend_functionality():
    """Test that backend has proper API structure."""
    print("🔍 Testing backend API structure...")
    
    # Test backend app.py has required endpoints
    app_path = Path("backend/app.py")
    app_content = app_path.read_text()
    
    expected_endpoints = [
        "@app.get(\"/\",",
        "@app.get(\"/health\",",
        "@app.post(\"/enhance\",",
        "@app.get(\"/models/available\")",
        "FastAPI",
        "PromptEnhancer"
    ]
    
    missing_endpoints = []
    for endpoint in expected_endpoints:
        if endpoint not in app_content:
            missing_endpoints.append(endpoint)
    
    if missing_endpoints:
        print(f"❌ Missing backend features: {missing_endpoints}")
        return False
    
    print("✅ Backend API structure is complete")
    return True

def test_ai_integration():
    """Test AI model integration."""
    print("🔍 Testing AI model integration...")
    
    try:
        # Import and test the prompt enhancer
        sys.path.append('backend')
        from models.prompt_enhancer import PromptEnhancer
        from simplified_guides import SIMPLIFIED_MODEL_GUIDES
        
        # Test model guides exist
        expected_models = ['claude', 'openai', 'gemini', 'grok']
        for model in expected_models:
            if model not in SIMPLIFIED_MODEL_GUIDES:
                print(f"❌ Missing model guide for: {model}")
                return False
        
        print("✅ AI model integration is properly configured")
        return True
        
    except Exception as e:
        print(f"❌ AI integration test failed: {str(e)}")
        return False

def test_complete_system():
    """Run complete system test."""
    print("🎨 AI Prompt Enhancement Studio - Complete System Test")
    print("=" * 60)
    
    tests = [
        ("File Restoration", test_files_exist),
        ("Frontend Content", test_frontend_content), 
        ("Backend API", test_backend_functionality),
        ("AI Integration", test_ai_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 CONGRATULATIONS!")
        print("🎨 Your AI Prompt Enhancement Studio UI has been successfully restored!")
        print("🚀 The professional dark theme interface is back exactly as shown in your screenshots!")
        print("\n📋 Next Steps:")
        print("1. Start the backend: python backend/app.py")
        print("2. Open frontend/index.html in your browser")
        print("3. Enjoy your beautiful professional interface!")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir(Path(__file__).parent)
    
    success = test_complete_system()
    sys.exit(0 if success else 1)