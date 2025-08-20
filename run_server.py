"""
Startup script for the Airspace Traffic Analysis API server.

This script starts the FastAPI server with the appropriate configuration.
"""

import sys
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    from server_tailwind.main import app
    
    print("Starting Airspace Traffic Analysis API server...")
    print("API will be available at: http://localhost:8000")
    print("Interactive API docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Enable auto-reload for development
        log_level="info"
    )