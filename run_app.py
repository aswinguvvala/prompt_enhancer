#!/usr/bin/env python3
"""
Simple startup script for AI Prompt Enhancement Studio
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def main():
    print("🎨 AI Prompt Enhancement Studio")
    print("=" * 40)
    
    # Get current directory
    project_dir = Path(__file__).parent
    backend_dir = project_dir / "backend"
    frontend_file = project_dir / "frontend" / "index.html"
    
    print(f"📁 Project directory: {project_dir}")
    print(f"🔧 Backend directory: {backend_dir}")
    print(f"🌐 Frontend file: {frontend_file}")
    
    # Check if backend directory exists
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return
        
    # Check if frontend file exists
    if not frontend_file.exists():
        print("❌ Frontend file not found!")
        return
    
    print("\n🚀 Starting backend server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("📊 System Status: http://localhost:8000/status")
    
    # Set Python path and start server
    env = os.environ.copy()
    env['PYTHONPATH'] = str(backend_dir)
    
    try:
        # Start the server
        print("\n⚡ Server starting on http://localhost:8000")
        print("Press Ctrl+C to stop\n")
        
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend.app:app",
            "--host", "localhost",
            "--port", "8000",
            "--reload"
        ], env=env, cwd=str(project_dir))
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n💡 You can also try:")
        print("   1. Open terminal in project directory")
        print("   2. Run: PYTHONPATH=backend python -m uvicorn backend.app:app --host localhost --port 8000 --reload")
        print(f"   3. Open: {frontend_file}")

if __name__ == "__main__":
    main()