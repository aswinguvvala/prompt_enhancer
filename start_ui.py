#!/usr/bin/env python3
"""
Easy startup script for AI Prompt Enhancement Studio with beautiful UI
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def start_server():
    """Start the FastAPI backend server."""
    print("🚀 Starting AI Prompt Enhancement Studio backend...")
    
    # Change to the correct directory
    os.chdir(Path(__file__).parent)
    
    # Start the FastAPI server
    try:
        # Start backend on port 8001 (to avoid conflicts)
        env = os.environ.copy()
        env['API_PORT'] = '8001'
        
        process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 
            'backend.app:app',
            '--host', 'localhost',
            '--port', '8001',
            '--reload'
        ], env=env)
        
        print("✅ Backend server starting on http://localhost:8001")
        print("⏳ Waiting for server to be ready...")
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open the frontend in the browser
        frontend_url = "http://localhost:8001/"
        print(f"🌐 Opening your beautiful AI Prompt Studio at: {frontend_url}")
        
        try:
            webbrowser.open(frontend_url)
        except Exception as e:
            print(f"Could not auto-open browser: {e}")
            print(f"Please manually open: {frontend_url}")
        
        print("\n🎨 Your professional AI Prompt Enhancement Studio is ready!")
        print("✨ Features restored:")
        print("  • Professional dark theme with gradients")
        print("  • Model selection cards (OpenAI, Claude, Gemini, Grok)")
        print("  • Real AI-powered prompt enhancement")
        print("  • Beautiful before/after comparison")
        print("  • Copy and export functionality")
        print("  • Professional typography and animations")
        
        print("\n📋 Usage:")  
        print("  1. Select your target AI model")
        print("  2. Enter your prompt to enhance")
        print("  3. Click 'Enhance with AI Intelligence'")
        print("  4. See the professional enhancement results!")
        
        print("\n🔧 Controls:")
        print("  • Ctrl+Enter: Quick enhance")
        print("  • Escape: Clear all")
        
        print(f"\n🌐 Access your studio at: {frontend_url}")
        print("Press Ctrl+C to stop the server")
        
        # Keep the server running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down AI Prompt Enhancement Studio...")
            process.terminate()
            process.wait()
            print("✅ Server stopped successfully")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Make sure you're in the project directory")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Set OpenAI API key: export OPENAI_API_KEY='your_key'")
        return False
    
    return True

if __name__ == "__main__":
    print("🎨 AI Prompt Enhancement Studio - Professional UI Launcher")
    print("=" * 60)
    
    success = start_server()
    sys.exit(0 if success else 1)