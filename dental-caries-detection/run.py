"""Run the dental caries detection web application."""

import os
from app.main import create_app
from app.core.config import Config
from flask_cors import CORS

if __name__ == '__main__':
    # Create necessary directories
    Config.setup()
    
    # Create and configure the application
    app = create_app()
    CORS(app)  # Enable CORS for all routes
    
    # Run the application
    app.run(
        host='0.0.0.0',  # Make server publicly available
        port=5000,       # Port to run the server on
        debug=True       # Enable debug mode for development
    ) 