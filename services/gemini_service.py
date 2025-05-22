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
        
        # For conversation summarization
        self.conversation_summaries = []
        self.messages_since_last_summary = 0
        self.summary_threshold = 10  # Summarize every 10 messages
        
        # For LLM-based summarization
        self.llm_summaries = []
        self.full_history = []
    
    def reset_chat(self):
        """Reset the chat session but keep the model initialized"""
        self.chat = None
        self.chat_initialized = False
        self.all_sessions_history = []
        self.using_consolidated_history = False
        self.llm_summaries = []
        self.full_history = []
    
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
        
        # Generate an initial LLM-based summary of all loaded histories
        if len(self.all_sessions_history) > 15:  # Only summarize if we have substantial history
            self.llm_summaries.append(self.generate_llm_summary(self.all_sessions_history))
        
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
            
        # Add the current query to full history
        user_message = {"role": "user", "content": query}
        self.full_history.append(user_message)
        
        # Check if we need to generate a new summary based on history length
        if len(self.full_history) % 15 == 0 and len(self.full_history) > 15:
            summary = self.generate_llm_summary(self.full_history)
            self.llm_summaries.append(summary)
        
        try:
            # Initialize chat session if not already done
            if not self.chat_initialized:
                # Start with the system prompt alone
                self.chat = self.model.start_chat(history=[])
                self.chat.send_message(self.system_prompt)
                
                # Process history before current query
                if history:
                    # Check if we should use summarized history instead of full history
                    if len(history) > 30 and self.llm_summaries:
                        # Create a message with all summaries plus recent interactions
                        summary_msg = "Here's a summary of our previous movie conversations:\n\n"
                        
                        # Add all generated summaries
                        for idx, summary in enumerate(self.llm_summaries):
                            summary_msg += f"Summary {idx+1}:\n{summary}\n\n"
                            
                        # Add the most recent interactions (last 10 messages) for immediate context
                        summary_msg += "\nOur most recent conversation:\n"
                        recent_messages = history[-10:] if len(history) > 10 else history
                        for message in recent_messages:
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
                        except Exception as summary_error:
                            print(f"Error sending summary history: {summary_error}")
                            # Fallback to an even more condensed version
                            self.chat.send_message(self.extract_user_preferences(history))
                    else:
                        # Use regular history approach for smaller histories
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
                            # If the history is too large, use the extracted preferences
                            print(f"Error sending full history: {history_error}")
                            print("Trying with condensed preferences...")
                            self.chat.send_message(self.extract_user_preferences(history))
                
                self.chat_initialized = True
            
            # Send the current query and get response
            response = self.chat.send_message(query)
            
            # Add the assistant's response to full history
            assistant_message = {"role": "assistant", "content": response.text}
            self.full_history.append(assistant_message)
            
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
                
                # Determine if we should use summaries instead of full history
                if len(history) > 30 and self.llm_summaries:
                    full_prompt = self.system_prompt + "\n\n"
                    full_prompt += "Here are summaries of our previous conversations:\n\n"
                    
                    # Add all generated summaries
                    for idx, summary in enumerate(self.llm_summaries):
                        full_prompt += f"Summary {idx+1}:\n{summary}\n\n"
                        
                    # Add recent messages for immediate context
                    full_prompt += "\nRecent conversation:\n"
                    recent_messages = history[-5:] if len(history) > 5 else history
                    for message in recent_messages:
                        role = message.get("role")
                        content = message.get("content")
                        if role == "user":
                            full_prompt += f"User said: {content}\n"
                        elif role == "assistant":
                            full_prompt += f"You responded: {content}\n"
                else:
                    # Use the traditional approach for smaller histories
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
                
                # Add the assistant's response to full history
                assistant_message = {"role": "assistant", "content": response.text}
                self.full_history.append(assistant_message)
                
                return response.text
                
            except Exception as nested_e:
                return f"Error getting response: {str(nested_e)}"
    
    def generate_llm_summary(self, history: List[Dict[str, str]]) -> str:
        """
        Generate a summary of the conversation using the LLM itself
        
        Args:
            history: List of message dictionaries in the conversation
            
        Returns:
            A concise summary of the conversation highlighting key points and user preferences
        """
        try:
            # Prepare a prompt specifically for summarization
            summary_prompt = """
            Please create a concise summary of the following conversation about movies.
            Focus on capturing:
            1. The user's movie preferences (genres, directors, actors they like/dislike)
            2. Any specific movies the user mentioned enjoying or not enjoying
            3. Key questions the user asked
            4. Important information or recommendations you provided
            
            Make the summary informative enough that it could be used to continue the conversation
            while understanding the user's preferences, but keep it concise.
            
            Conversation:
            """
            
            # Add the conversation to summarize
            for message in history:
                role = message.get("role", "")
                content = message.get("content", "")
                
                # Skip system or marker messages
                if role not in ["user", "assistant"] or ("Session" in content and "History" in content):
                    continue
                    
                # Add the message to be summarized
                summary_prompt += f"\n{role.capitalize()}: {content}"
            
            # Generate the summary using the LLM
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more focused summary
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,  # Limit output length
            }
            
            response = self.model.generate_content(
                summary_prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            # Fallback to simple summarization
            return self.summarize_conversation(history)
    
    def extract_user_preferences(self, history: List[Dict[str, str]]) -> str:
        """
        Extract just the user preferences from the conversation history
        
        Args:
            history: The conversation history
            
        Returns:
            A string containing extracted user preferences
        """
        preferences_msg = "From our previous conversations, I've learned the following about your movie preferences:\n\n"
        
        # Add only user messages that likely contain preferences
        pref_keywords = ["like", "love", "hate", "dislike", "enjoy", "favorite", "prefer", 
                         "genre", "director", "actor", "similar", "recommend"]
        
        # Track if we found any preferences
        found_preferences = False
        
        for message in history:
            if message.get("role") == "user":
                content = message.get("content", "").lower()
                # Check if message likely contains preferences
                if any(keyword in content for keyword in pref_keywords):
                    preferences_msg += f"- {message.get('content')}\n"
                    found_preferences = True
                    
        # If no explicit preferences were found, use the first few and last few user messages
        if not found_preferences and history:
            preferences_msg = "Here are some key points from our previous conversations:\n\n"
            
            # Get user messages
            user_messages = [msg for msg in history if msg.get("role") == "user"]
            
            # Take up to 3 from beginning and end if available
            important_msgs = []
            if len(user_messages) <= 6:
                important_msgs = user_messages
            else:
                important_msgs = user_messages[:3] + user_messages[-3:]
                
            for msg in important_msgs:
                preferences_msg += f"- {msg.get('content')}\n"
                
        return preferences_msg

    def summarize_conversation(self, history: List[Dict[str, str]]) -> str:
        """
        Create a simple extractive summary of the conversation history (fallback method)
        
        Args:
            history: List of messages in the conversation
            
        Returns:
            Summary of the conversation
        """
        try:
            # Simple extractive summarization: take the first and last N messages, and important middle messages
            N = 3  # Number of messages to take from beginning and end
            important_messages = history[:N] + history[-N:]
            
            # Add some contextual messages around the important ones
            if len(history) > 2 * N:
                important_messages = history[N:-N] + important_messages
            
            # Generate a summary from the selected messages
            summary = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in important_messages])
            return f"Summary of our conversation:\n{summary}"
        
        except Exception as e:
            return f"Error summarizing conversation: {str(e)}"
    
    def add_message_to_summary(self, message: Dict[str, str]):
        """
        Add a message to the conversation summary
        
        Args:
            message: The message to add (should be in the format used by the Gemini API)
        """
        try:
            self.messages_since_last_summary += 1
            
            # Add the message to the summary history
            self.conversation_summaries.append(message)
            
            # If we've reached the threshold, summarize the conversation
            if self.messages_since_last_summary >= self.summary_threshold:
                summary = self.generate_llm_summary(self.conversation_summaries)
                
                # Reset the summary history
                self.conversation_summaries = []
                self.messages_since_last_summary = 0
                
                # Add the summary as a message in the conversation
                self.conversation_summaries.append({
                    "role": "assistant",
                    "content": summary
                })
                
        except Exception as e:
            print(f"Error adding message to summary: {str(e)}")