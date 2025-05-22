"""
API initialization module for both Flask and FastAPI
"""

def create_app():
    """Create and configure the Flask application"""
    from flask import Flask
    app = Flask(__name__)
    
    # Import routes after app is created to avoid circular imports
    from app.api.routes import bp as routes_bp
    app.register_blueprint(routes_bp)
    
    return app

def create_fast_api():
    """Create and configure the FastAPI application"""
    from fastapi import FastAPI
    
    fast_api_app = FastAPI(
        title="Movie Recommendation API",
        description="API for movie recommendations using Gemini LLM",
        version="1.0.0"
    )
    
    # Import FastAPI routes
    from app.api.fast_api_routes import router
    fast_api_app.include_router(router)
    
    return fast_api_app

__all__ = ["create_app", "create_fast_api"]