"""
Flask routes for the movie recommendation API
"""
from flask import Blueprint, request, jsonify
import os

from services.gemini_service import GeminiService
from models.schemas import ChatbotRequest, ChatbotResponse

bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize the Gemini service
gemini_service = GeminiService()

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
        # Get response from Gemini
        response = gemini_service.get_movie_recommendation(query, history)
        
        # Update conversation history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})
        
        return jsonify({
            'response': response,
            'conversation_history': history
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500