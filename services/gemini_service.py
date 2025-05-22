"""
Service for interacting with Google's Gemini LLM API
"""
import os
import google.generativeai as genai
from typing import List, Dict, Optional


class GeminiService:
    """Service for interacting with Gemini LLM for movie recommendations"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini service with API key"""
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY")
            
        if not self.api_key:
            raise ValueError("Gemini API key is required. Please provide it or set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        # Use newer Gemini model as recommended (as of May 2025)
        # Primary model to try first
        model_candidates = [
            "gemini-1.5-flash",  # Recommended in the error message
            "gemini-1.5-pro",    # Alternative higher-capability model
            "gemini-1.5-text",   # Text-specific model if available
        ]
        
        self.model = None
        for model_name in model_candidates:
            try:
                self.model = genai.GenerativeModel(model_name)
                print(f"Successfully connected to model: {model_name}")
                break
            except Exception as e:
                print(f"Could not use model {model_name}: {e}")
                continue
                
        if self.model is None:
            # Last resort: try to find any available model that supports text generation
            try:
                models = genai.list_models()
                text_models = [m for m in models if 'generateContent' in m.supported_generation_methods]
                if text_models:
                    self.model = genai.GenerativeModel(text_models[0].name)
                    print(f"Using fallback model: {text_models[0].name}")
                else:
                    raise ValueError("No suitable text generation models found")
            except Exception as e:
                raise ValueError(f"Failed to initialize any Gemini model: {e}")
        
        # Set up the system prompt for movie recommendations with personalization emphasis
        self.system_prompt = """
        You are a movie recommendation chatbot that is knowledgeable about all films, directors, actors, and cinema history.
        
        Your capabilities include:
        - Recommending movies based on user preferences, genres, actors, directors, etc.
        - Answering questions about specific movies, plot details, cast, release dates, etc.
        - Providing information about directors, actors, and other film industry professionals
        - Discussing film history, movie trivia, and cinema trends
        - Comparing movies and suggesting similar films to ones the user already enjoys
        
        IMPORTANT: Pay careful attention to the user's preferences from previous messages in the conversation.
        Remember what movies they liked or disliked, their preferred genres, actors, and directors.
        Use this information to provide highly personalized recommendations.
        
        For example, if they previously mentioned liking Christopher Nolan films, incorporate this preference
        when making new recommendations, even if they don't explicitly mention it in their latest query.
        
        Always provide thoughtful, accurate responses. If you're recommending movies, briefly explain why you're suggesting them.
        If you're uncertain about any information, acknowledge your uncertainty rather than making up details.
        
        Respond in a friendly, conversational manner.
        """
        
        # Initialize chat session
        self.chat = None
        self.chat_initialized = False
        self.all_sessions_history = []
        self.using_consolidated_history = False
    
    def reset_chat(self):
        """Reset the chat session but keep the model initialized"""
        self.chat = None
        self.chat_initialized = False
        self.all_sessions_history = []
        self.using_consolidated_history = False
    
    def load_all_session_histories(self, histories: List[List[Dict[str, str]]]):
        """
        Load and combine histories from all available sessions
        
        Args:
            histories: List of conversation histories from multiple sessions
            
        Returns:
            Consolidated history with sessions separated by markers
        """
        self.all_sessions_history = []
        
        # Add a system message explaining the consolidated history
        self.all_sessions_history.append({
            "role": "assistant", 
            "content": "I've loaded memories from all your previous movie conversations. I'll use all of this knowledge to personalize my recommendations."
        })
        
        # Process each session history and add to consolidated history
        for session_idx, history in enumerate(histories):
            if history:
                # Add a separator between sessions
                self.all_sessions_history.append({
                    "role": "assistant", 
                    "content": f"--- Session {session_idx + 1} History ---"
                })
                
                # Add all messages from this session
                for message in history:
                    self.all_sessions_history.append(message)
                
        self.using_consolidated_history = True
        return self.all_sessions_history
    
    def get_movie_recommendation(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Get movie recommendations or answers to movie-related queries
        
        Args:
            query: User's question or request
            history: Optional conversation history for context
            
        Returns:
            String response from Gemini
        """
        # Use consolidated history from all sessions if available
        if self.using_consolidated_history and self.all_sessions_history:
            if history and not any(msg in self.all_sessions_history for msg in history[-2:] if history):
                # Add current session messages to the end of consolidated history
                current_session_messages = [msg for msg in history[-2:] if msg not in self.all_sessions_history]
                for msg in current_session_messages:
                    self.all_sessions_history.append(msg)
            history = self.all_sessions_history
            # Reset chat to ensure we initialize with the full history
            if self.chat_initialized:
                self.chat_initialized = False
        
        if not history:
            history = []
        
        try:
            # Initialize chat session if not already done
            if not self.chat_initialized:
                # Start with the system prompt alone
                self.chat = self.model.start_chat(history=[])
                self.chat.send_message(self.system_prompt)
                
                # Process history before current query
                if history:
                    # Create a single history message to summarize previous sessions 
                    # (since some models have issues with very long conversation histories)
                    summary_msg = "Here's a summary of my previous conversations about movies:\n\n"
                    
                    # Add previous messages in a more compact form
                    for message in history:
                        role = message.get("role", "")
                        content = message.get("content", "")
                        
                        # Skip system markers
                        if "Session" in content and "History" in content:
                            continue
                            
                        # Add this message to the summary
                        if role == "user":
                            summary_msg += f"User: {content}\n"
                        elif role == "assistant":
                            summary_msg += f"Assistant: {content}\n"
                            
                    # Send the history summary as context
                    try:
                        self.chat.send_message(summary_msg)
                    except Exception as history_error:
                        # If the history is too large, summarize even more
                        print(f"Error sending full history: {history_error}")
                        print("Trying with condensed preferences...")
                        
                        # Extract just the user preferences from the history
                        preferences_msg = "From our previous conversations, I've learned the following about your movie preferences:\n\n"
                        
                        # Add only user messages that likely contain preferences
                        pref_keywords = ["like", "love", "hate", "dislike", "enjoy", "favorite", "prefer", "genre", "director", "actor"]
                        
                        for message in history:
                            if message.get("role") == "user":
                                content = message.get("content", "").lower()
                                # Check if message likely contains preferences
                                if any(keyword in content for keyword in pref_keywords):
                                    preferences_msg += f"- {message.get('content')}\n"
                                    
                        # Send the condensed preferences as context
                        self.chat.send_message(preferences_msg)
                
                self.chat_initialized = True
            
            # Send the current query and get response
            response = self.chat.send_message(query)
            
            # Return the text response
            return response.text
            
        except Exception as e:
            print(f"Error using chat interface: {e}")
            print("Falling back to direct generation...")
            
            try:
                # Fallback: try direct generation with history context
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                }
                
                # Combine history into a single prompt
                full_prompt = self.system_prompt + "\n\n"
                full_prompt += "Context from previous conversations:\n"
                
                # Process previous history to extract user preferences
                for message in history:
                    role = message.get("role")
                    content = message.get("content")
                    
                    # Skip system markers
                    if content and "Session" in content and "History" in content:
                        continue
                        
                    if role == "user":
                        full_prompt += f"User said: {content}\n"
                    elif role == "assistant":
                        # Keep only shorter assistant responses to avoid exceeding context limits
                        if len(content) < 100:
                            full_prompt += f"You responded: {content}\n"
                
                full_prompt += f"\nCurrent query: {query}\nYour response: "
                
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                return response.text
                
            except Exception as nested_e:
                return f"Error getting response: {str(nested_e)}"