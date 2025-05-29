"""
Pydantic models for the movie recommendation chatbot
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MovieQuery(BaseModel):
    """Represents a movie recommendation query from the user"""
    query: str = Field(..., description="The user's query about movies")


class MovieRecommendation(BaseModel):
    """Represents a movie recommendation response"""
    title: str = Field(..., description="Movie title")
    year: Optional[int] = Field(None, description="Release year")
    director: Optional[str] = Field(None, description="Movie director")
    genre: Optional[List[str]] = Field(None, description="Movie genres")
    description: Optional[str] = Field(None, description="Brief description of the movie")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "The Shawshank Redemption",
                "year": 1994,
                "director": "Frank Darabont",
                "genre": ["Drama"],
                "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."
            }
        }


class ConversationHistory(BaseModel):
    """Represents the conversation history between user and chatbot"""
    messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of conversation messages with 'role' and 'content'"
    )


class Intent(BaseModel):
    """Represents a recognized user intent"""
    intent: str = Field(..., description="The recognized intent type")
    confidence: float = Field(..., description="Confidence score for the intent (0-1)")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Entities extracted from the user query (e.g., movie titles, actor names)"
    )
    all_intents: Optional[Dict[str, float]] = Field(
        None, 
        description="All detected intents with their confidence scores"
    )


class ChatbotRequest(BaseModel):
    """Request model for chatbot API"""
    query: str = Field(..., description="User query text")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list,
        description="Previous conversation history"
    )


class ChatbotResponse(BaseModel):
    """Response model for chatbot API"""
    response: str = Field(..., description="Chatbot response text")
    conversation_history: List[Dict[str, str]] = Field(
        ..., 
        description="Updated conversation history"
    )
    intent: Optional[Intent] = Field(
        None,
        description="Recognized intent from the user's query"
    )