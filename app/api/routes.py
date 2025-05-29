"""
Flask routes for the movie recommendation API
"""
from flask import Blueprint, request, jsonify
import os

from services.gemini_service import GeminiService
from services.intent_service import IntentService
from models.schemas import ChatbotRequest, ChatbotResponse, Intent

bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize the Gemini service
gemini_service = GeminiService()

# Initialize the Intent service with Gemini service
intent_service = IntentService(gemini_service=gemini_service)

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'movie-recommendation-chatbot'})

@bp.route('/chat', methods=['POST'])
def chat():
    """Endpoint for chatting with the movie recommendation bot"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
    
    query = data['query']
    history = data.get('conversation_history', [])
    
    try:
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
        history.append({"role": "user", "content": history_query})
        history.append({"role": "assistant", "content": response})
        
        return jsonify({
            'response': response,
            'conversation_history': history,
            'intent': intent_result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/intent', methods=['POST'])
def analyze_intent():
    """Endpoint for analyzing user intent only"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
    
    query = data['query']
    history = data.get('conversation_history', [])
    
    try:
        # Recognize intent using the hybrid approach
        intent_result = intent_service.recognize_intent(query, history)
        
        return jsonify(intent_result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500