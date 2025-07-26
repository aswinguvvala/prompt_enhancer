#!/usr/bin/env python3
"""
Test script to verify the AI Prompt Enhancement Studio server
"""

import os
import sys
import asyncio
import time

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_server():
    """Test server functionality"""
    print("🧪 Testing AI Prompt Enhancement Studio Server")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("1. Testing imports...")
        from config import settings
        print(f"   ✅ Settings loaded - AI Backend: {settings.AI_BACKEND}")
        
        from models.prompt_enhancer import PromptEnhancer
        print("   ✅ PromptEnhancer imported successfully")
        
        from models.main_llm import model_manager
        print("   ✅ ModelManager imported successfully")
        
        from models.evaluator import ai_evaluator
        print("   ✅ AIEvaluator imported successfully")
        
        # Test prompt enhancer initialization
        print("\n2. Testing PromptEnhancer initialization...")
        enhancer = PromptEnhancer()
        print(f"   ✅ Primary backend: {enhancer.primary_backend}")
        print(f"   ✅ Fallback backend: {enhancer.fallback_backend}")
        
        # Test health check
        print("\n3. Testing backend health...")
        health_status = await enhancer.health_check()
        for backend, status in health_status.items():
            if isinstance(status, dict) and 'available' in status:
                available = "✅" if status['available'] else "❌"
                print(f"   {available} {backend}: {status.get('available', False)}")
        
        # Test basic enhancement (with fallback)
        print("\n4. Testing basic prompt enhancement...")
        test_prompt = "Write a short story about a robot."
        context_prompt = f"You are a helpful assistant. Please enhance this prompt: {test_prompt}"
        
        try:
            result = await enhancer.enhance_prompt(
                context_injection_prompt=context_prompt,
                original_prompt=test_prompt,
                target_model="openai",
                temperature=0.7
            )
            print(f"   ✅ Enhancement completed")
            print(f"   📝 Original: {result.original_prompt[:50]}...")
            print(f"   🚀 Enhanced: {result.enhanced_prompt[:50]}...")
            print(f"   ⚡ Backend used: {result.backend_used}")
            print(f"   ⏱️  Processing time: {result.processing_time:.2f}s")
            
        except Exception as e:
            print(f"   ⚠️  Enhancement failed (expected if no AI backends available): {str(e)}")
        
        print("\n5. Starting FastAPI server...")
        
        # Import and start the FastAPI app
        from app import app
        import uvicorn
        
        print("   ✅ FastAPI app imported successfully")
        print("   🚀 Starting server on http://localhost:8000")
        print("   📖 API docs available at http://localhost:8000/docs")
        print("   🏥 Health check at http://localhost:8000/health")
        print("   📊 Status endpoint at http://localhost:8000/status")
        print("\n   Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            app, 
            host="localhost", 
            port=8000, 
            log_level="info",
            reload=False  # Disable reload for testing
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_server())