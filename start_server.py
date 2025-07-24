#!/usr/bin/env python3
"""
AI Prompt Enhancement Studio - Development Server Startup Script
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("📦 Please install dependencies with: pip install -r requirements.txt")
        return False

def start_backend_server():
    """Start the FastAPI backend server."""
    print("🚀 Starting AI Prompt Enhancement Studio backend server...")
    
    backend_path = Path(__file__).parent / "backend"
    os.chdir(backend_path)
    
    try:
        # Start the server using uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "localhost",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def open_frontend():
    """Open the frontend in the default browser."""
    time.sleep(2)  # Wait for server to start
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    
    if frontend_path.exists():
        webbrowser.open(f"file://{frontend_path.absolute()}")
        print(f"🌐 Opening frontend: {frontend_path.absolute()}")
    else:
        print("❌ Frontend file not found")

def main():
    """Main startup function."""
    print("=" * 60)
    print("🎨 AI Prompt Enhancement Studio")
    print("Multi-Model AI Prompt Optimization System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Display system information
    print("\n📊 System Information:")
    print(f"   • Python Version: {sys.version.split()[0]}")
    print(f"   • Working Directory: {Path.cwd()}")
    print(f"   • Backend API: http://localhost:8000")
    print(f"   • API Documentation: http://localhost:8000/docs")
    print(f"   • System Status: http://localhost:8000/status")
    
    print("\n🤖 Available AI Models:")
    models = [
        ("OpenAI GPT-4", "Advanced reasoning and creativity"),
        ("Anthropic Claude", "Helpful, harmless, and honest"),
        ("Google Gemini", "Multimodal AI capabilities"),
        ("xAI Grok", "Real-time data and reasoning")
    ]
    
    for name, description in models:
        print(f"   • {name}: {description}")
    
    print("\n🚀 Starting servers...")
    
    # Start backend server
    try:
        start_backend_server()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()