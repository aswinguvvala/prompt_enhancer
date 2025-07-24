#!/usr/bin/env python3
"""
AI Prompt Enhancement Studio - System Integration Tests
"""

import asyncio
import json
import time
from typing import Dict, Any
import sys
from pathlib import Path

# Add backend to path for testing
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from backend.app import MODEL_RULES, PromptOptimizer, PromptRequest
    print("✅ Successfully imported backend modules")
except ImportError as e:
    print(f"❌ Failed to import backend modules: {e}")
    sys.exit(1)

class SystemTester:
    """Comprehensive system testing suite."""
    
    def __init__(self):
        self.optimizer = PromptOptimizer()
        self.test_results = []
    
    def run_all_tests(self):
        """Run all system tests."""
        print("🧪 Starting AI Prompt Enhancement Studio Integration Tests")
        print("=" * 60)
        
        tests = [
            ("Model Rules Validation", self.test_model_rules),
            ("Prompt Optimization", self.test_prompt_optimization),
            ("Quality Evaluation", self.test_quality_evaluation),
            ("Content Generation", self.test_content_generation),
            ("Model-Specific Enhancements", self.test_model_specific_enhancements),
            ("Enhancement Types", self.test_enhancement_types),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            try:
                result = test_func()
                if result:
                    print(f"   ✅ PASSED")
                    passed += 1
                else:
                    print(f"   ❌ FAILED")
                self.test_results.append((test_name, result))
            except Exception as e:
                print(f"   💥 ERROR: {e}")
                self.test_results.append((test_name, False))
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! System is ready for use.")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
        
        return passed == total
    
    def test_model_rules(self) -> bool:
        """Test that all model rules are properly defined."""
        required_models = ["openai", "claude", "gemini", "grok"]
        
        for model_id in required_models:
            if model_id not in MODEL_RULES:
                print(f"   ❌ Missing model: {model_id}")
                return False
            
            model_data = MODEL_RULES[model_id]
            if not model_data.get("name"):
                print(f"   ❌ Missing name for model: {model_id}")
                return False
            
            if not model_data.get("rules"):
                print(f"   ❌ Missing rules for model: {model_id}")
                return False
            
            if not model_data.get("enhancements"):
                print(f"   ❌ Missing enhancements for model: {model_id}")
                return False
        
        print(f"   ✓ All {len(required_models)} models properly configured")
        return True
    
    def test_prompt_optimization(self) -> bool:
        """Test prompt optimization functionality."""
        test_prompt = "Write a story about AI"
        
        for model_id in MODEL_RULES.keys():
            enhanced = self.optimizer.apply_model_enhancements(test_prompt, model_id)
            
            if enhanced == test_prompt:
                print(f"   ❌ No enhancement applied for {model_id}")
                return False
            
            if len(enhanced) <= len(test_prompt):
                print(f"   ❌ Enhancement didn't add content for {model_id}")
                return False
        
        print(f"   ✓ Prompt optimization working for all {len(MODEL_RULES)} models")
        return True
    
    def test_quality_evaluation(self) -> bool:
        """Test prompt quality evaluation."""
        test_prompts = [
            "Hi",  # Low quality
            "Please write a detailed analysis of machine learning algorithms with examples and use cases",  # High quality
            "Create a comprehensive guide explaining the principles of quantum computing"  # High quality
        ]
        
        for prompt in test_prompts:
            quality = self.optimizer.evaluate_prompt_quality(prompt)
            
            # Check that all metrics are present and within valid range
            required_metrics = ["specificity", "clarity", "completeness", "actionability", "overall"]
            for metric in required_metrics:
                if not hasattr(quality, metric):
                    print(f"   ❌ Missing quality metric: {metric}")
                    return False
                
                value = getattr(quality, metric)
                if not (0 <= value <= 1):
                    print(f"   ❌ Invalid metric value: {metric} = {value}")
                    return False
        
        print("   ✓ Quality evaluation working correctly")
        return True
    
    def test_content_generation(self) -> bool:
        """Test content generation for different types."""
        enhancement_types = ["general", "creative_writing", "technical", "analysis", "coding"]
        test_prompt = "Enhanced: Explain machine learning concepts"
        
        for enhancement_type in enhancement_types:
            content = self.optimizer.generate_mock_content(test_prompt, enhancement_type)
            
            if not content or len(content) < 50:
                print(f"   ❌ Insufficient content generated for {enhancement_type}")
                return False
            
            # Check for type-specific content
            if enhancement_type == "coding" and "```" not in content:
                print(f"   ❌ Code blocks missing in coding content")
                return False
            
            if enhancement_type == "technical" and "##" not in content:
                print(f"   ❌ Headers missing in technical content")
                return False
        
        print(f"   ✓ Content generation working for all {len(enhancement_types)} types")
        return True
    
    def test_model_specific_enhancements(self) -> bool:
        """Test model-specific enhancement features."""
        test_prompt = "Analyze the data"
        
        # Test Claude XML tags
        claude_enhanced = self.optimizer.apply_model_enhancements(test_prompt, "claude")
        if "<instructions>" not in claude_enhanced:
            print("   ❌ Claude XML tags not applied")
            return False
        
        # Test Grok real-time data
        grok_enhanced = self.optimizer.apply_model_enhancements(test_prompt, "grok")
        if "current information" not in grok_enhanced.lower():
            print("   ❌ Grok real-time data prefix not applied")
            return False
        
        # Test OpenAI structure
        openai_enhanced = self.optimizer.apply_model_enhancements(test_prompt, "openai")
        if "expert assistant" not in openai_enhanced.lower():
            print("   ❌ OpenAI expert role not applied")
            return False
        
        # Test Gemini natural language
        gemini_enhanced = self.optimizer.apply_model_enhancements(test_prompt, "gemini")
        if "expert in this domain" not in gemini_enhanced.lower():
            print("   ❌ Gemini domain expertise not applied")
            return False
        
        print("   ✓ Model-specific enhancements working correctly")
        return True
    
    def test_enhancement_types(self) -> bool:
        """Test different enhancement types."""
        base_prompt = "Explain the concept"
        enhancement_types = ["general", "creative_writing", "technical", "analysis", "coding"]
        
        for enhancement_type in enhancement_types:
            enhanced = self.optimizer.apply_model_enhancements(base_prompt, "openai", enhancement_type)
            
            # Check for type-specific additions
            type_indicators = {
                "creative_writing": "creative",
                "technical": "technical",
                "analysis": "data-driven",
                "coding": "code"
            }
            
            if enhancement_type in type_indicators:
                if type_indicators[enhancement_type] not in enhanced.lower():
                    print(f"   ❌ Enhancement type {enhancement_type} not properly applied")
                    return False
        
        print(f"   ✓ All {len(enhancement_types)} enhancement types working")
        return True
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid inputs."""
        # Test invalid model
        result = self.optimizer.apply_model_enhancements("test", "invalid_model")
        if result != "test":
            print("   ❌ Invalid model not handled correctly")
            return False
        
        # Test empty prompt
        quality = self.optimizer.evaluate_prompt_quality("")
        if not (0 <= quality.overall <= 1):
            print("   ❌ Empty prompt quality evaluation failed")
            return False
        
        # Test very long prompt
        long_prompt = "test " * 1000
        enhanced = self.optimizer.apply_model_enhancements(long_prompt, "openai")
        if not enhanced:
            print("   ❌ Long prompt handling failed")
            return False
        
        print("   ✓ Error handling working correctly")
        return True

def main():
    """Run the system tests."""
    tester = SystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 System is ready! You can now start the server with:")
        print("   python start_server.py")
        print("\n🌐 Then open your browser to:")
        print("   Frontend: file:///.../frontend/index.html")
        print("   API Docs: http://localhost:8000/docs")
    else:
        print("\n🔧 Please fix the issues before running the system.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)