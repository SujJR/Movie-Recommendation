"""
FastAPI routes for the movie recommendation API
"""
from typing import List, Dict, Optional

# Initialize the Gemini service
gemini_service = None

def get_gemini_service():
    """Get or initialize the Gemini service"""
    global gemini_service
    if gemini_service is None:
        try:
            from services.gemini_service import GeminiService
            gemini_service = GeminiService()
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"Failed to initialize Gemini service: {str(e)}")
    return gemini_service

# Create router without importing FastAPI at module level
from fastapi import APIRouter, HTTPException, Depends
router = APIRouter(prefix="/api", tags=["Movie Recommendations"])

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "movie-recommendation-chatbot"}

@router.post("/chat")
async def chat(request_data: dict, service=Depends(get_gemini_service)):
    """Endpoint for chatting with the movie recommendation bot"""
    try:
        # Extract data from request
        query = request_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        history = request_data.get("conversation_history", [])
        
        # Get response from Gemini
        response = service.get_movie_recommendation(query, history)
        
        # Update conversation history
        history = list(history) if history else []
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})
        
        return {
            "response": response,
            "conversation_history": history
        }
    
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=str(e))
        raise