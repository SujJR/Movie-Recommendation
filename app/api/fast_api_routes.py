"""
FastAPI routes for the movie recommendation API
"""
from typing import List, Dict, Optional

# Define service singletons
gemini_service = None
intent_service = None

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

def get_intent_service():
    """Get or initialize the Intent service"""
    global intent_service, gemini_service
    if intent_service is None:
        try:
            # Make sure gemini service is initialized first
            if gemini_service is None:
                gemini_service = get_gemini_service()
                
            from services.intent_service import IntentService
            intent_service = IntentService(gemini_service=gemini_service)
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"Failed to initialize Intent service: {str(e)}")
    return intent_service

# Create router without importing FastAPI at module level
from fastapi import APIRouter, HTTPException, Depends
router = APIRouter(prefix="/api", tags=["Movie Recommendations"])

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "movie-recommendation-chatbot"}

@router.post("/chat")
async def chat(
    request_data: dict, 
    gemini_service=Depends(get_gemini_service),
    intent_service=Depends(get_intent_service)
):
    """Endpoint for chatting with the movie recommendation bot"""
    try:
        # Extract data from request
        query = request_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        history = request_data.get("conversation_history", [])
        
        # Recognize intent using the hybrid approach
        intent_result = intent_service.recognize_intent(query, history)
        
        # Check if the intent recognition resolved a list reference
        # For example, if the user said "Tell me about the 10th movie"
        if intent_result.get("resolved_reference") and "movie_title" in intent_result.get("entities", {}):
            # Modify the query to include the actual movie title for better LLM response
            movie_title = intent_result["entities"]["movie_title"][0]
            intent_type = intent_result["intent"]
            
            if intent_type == "plot":
                enhanced_query = f"Tell me the plot of the movie {movie_title}"
            elif intent_type == "actor":
                enhanced_query = f"Who are the actors in the movie {movie_title}?"
            elif intent_type == "director":
                enhanced_query = f"Who directed the movie {movie_title}?"
            elif intent_type == "rating":
                enhanced_query = f"What is the rating of the movie {movie_title}?"
            elif intent_type == "release":
                enhanced_query = f"When was the movie {movie_title} released?"
            else:
                enhanced_query = f"Tell me about the movie {movie_title}"
                
            # Use the enhanced query for the LLM
            response = gemini_service.get_movie_recommendation(enhanced_query, history)
            
            # But keep the original query in the conversation history
            history_query = query
        else:
            # Regular query processing
            response = gemini_service.get_movie_recommendation(query, history)
            history_query = query
        
        # Update conversation context with the assistant's response
        intent_service.update_conversation_context(response, intent_result.get("entities", {}))
        
        # Update conversation history
        history = list(history) if history else []
        history.append({"role": "user", "content": history_query})
        history.append({"role": "assistant", "content": response})
        
        return {
            "response": response,
            "conversation_history": history,
            "intent": intent_result
        }
    
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=str(e))
        raise

@router.post("/intent")
async def analyze_intent(
    request_data: dict,
    intent_service=Depends(get_intent_service)
):
    """Endpoint for analyzing user intent only"""
    try:
        # Extract data from request
        query = request_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        history = request_data.get("conversation_history", [])
        
        # Recognize intent using the hybrid approach
        intent_result = intent_service.recognize_intent(query, history)
        
        return intent_result
    
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=str(e))
        raise