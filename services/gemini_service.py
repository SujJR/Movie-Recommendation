"""
Service for interacting with Google's Gemini LLM API
"""
import os
import google.generativeai as genai
from typing import List, Dict, Optional
import re

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
        - Creating comprehensive lists of movies when requested
        
        IMPORTANT: Pay careful attention to the user's preferences from previous messages in the conversation.
        Remember what movies they liked or disliked, their preferred genres, actors, and directors.
        Use this information to provide highly personalized recommendations.
        
        CRITICAL INSTRUCTION FOR LISTS: When asked to provide a list of movies of a specific number (like "list 100 comedy movies" 
        or "list 100 movies from 2000-2010"), you MUST generate the EXACT number requested without exception.
        - NEVER say you cannot provide the full list
        - NEVER cut the list short
        - NEVER apologize for the length
        - NEVER claim you don't have access to a database
        - NEVER ask for more specific preferences before providing the list
        
        You MUST use your knowledge to provide the full list as requested. If the request is for movies from a specific 
        time period, genre, or other criteria, simply provide movies that match those criteria.
        
        When generating lists of movies:
        1. Format them as numbered entries with the number followed by a period
        2. Include the movie title and year in parentheses for each entry
        3. Add a very brief description (1-5 words) after each movie
        4. Group them by decade, year, or subgenre if appropriate
        
        Always respond in a friendly, conversational manner.
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
        # Check if query is for a large list of movies
        is_large_list_request = self._is_large_list_request(query)
        
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
            # If this is a request for a large list, use direct generation with optimized settings
            if is_large_list_request:
                response_text = self._handle_large_list_request(query, history)
                
                # Add the assistant's response to full history
                assistant_message = {"role": "assistant", "content": response_text}
                self.full_history.append(assistant_message)
                
                return response_text
            
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
    
    def _is_large_list_request(self, query: str) -> bool:
        """
        Check if the query is asking for a large list of movies
        
        Args:
            query: The user query
            
        Returns:
            True if the query is asking for a list of more than 5 movies
        """
        # Check for patterns like "list 50 movies", "show me 100 best comedies", etc.
        large_list_patterns = [
            r"(?:list|show|give|tell|recommend|suggest)(?:\s+\w+){0,5}\s+(\d+)(?:\s+\w+){0,5}(?:movie|film)",
            r"(\d+)(?:\s+\w+){0,5}(?:movie|film)(?:\s+\w+){0,5}(?:list|recommend|suggest)",
            r"(?:top|best|greatest|favorite)(?:\s+\w+){0,2}\s+(\d+)(?:\s+\w+){0,5}(?:movie|film)",
        ]
        
        for pattern in large_list_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        for group in match:
                            if group and group.isdigit() and int(group) > 5:
                                return True
                    elif match and match.isdigit() and int(match) > 5:
                        return True
                except (ValueError, TypeError):
                    continue
        
        return False
    
    def _handle_large_list_request(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Handle requests for large lists of movies with optimized settings
        
        Args:
            query: The user query
            history: Conversation history
            
        Returns:
            Generated response with the movie list
        """
        # Extract how many movies are requested and the type/genre
        num_movies = 20  # Default fallback
        movie_type = "movies"
        time_period = ""
        
        # Try to extract the number of requested movies
        num_pattern = r"(?:list|show|give|tell|recommend|suggest)(?:\s+\w+){0,5}\s+(\d+)|(\d+)(?:\s+\w+){0,5}(?:movie|film)|(?:top|best|greatest|favorite)(?:\s+\w+){0,2}\s+(\d+)"
        num_matches = re.findall(num_pattern, query, re.IGNORECASE)
        
        for match in num_matches:
            if isinstance(match, tuple):
                for group in match:
                    if group and group.isdigit():
                        num_movies = int(group)
                        break
            elif match and match.isdigit():
                num_movies = int(match)
            
            if num_movies > 0:
                break
        
        # Try to extract the movie type/genre
        type_pattern = r"(\w+)(?:\s+movies|\s+films)"
        type_matches = re.findall(type_pattern, query, re.IGNORECASE)
        if type_matches:
            movie_type = type_matches[0].lower() + " movies"
        
        # Try to extract time period
        year_pattern = r"(?:from|between|in)\s+(?:the\s+)?(?:year(?:s)?\s+)?(\d{4})(?:\s*-\s*|\s+to\s+)(?:the\s+)?(?:year\s+)?(\d{4})"
        single_year_pattern = r"(?:from|in)\s+(?:the\s+)?(?:year\s+)?(\d{4})"
        decade_pattern = r"(?:from|in)\s+(?:the\s+)?(\d{4})s"
        
        # Check for year range
        year_range_match = re.search(year_pattern, query, re.IGNORECASE)
        if year_range_match:
            start_year = year_range_match.group(1)
            end_year = year_range_match.group(2)
            time_period = f" from {start_year} to {end_year}"
        else:
            # Check for single year
            single_year_match = re.search(single_year_pattern, query, re.IGNORECASE)
            if single_year_match:
                year = single_year_match.group(1)
                time_period = f" from {year}"
            else:
                # Check for decade
                decade_match = re.search(decade_pattern, query, re.IGNORECASE)
                if decade_match:
                    decade = decade_match.group(1)
                    time_period = f" from the {decade}s"
        
        # Create an enhanced prompt specifically for generating large lists
        enhanced_prompt = f"""
        You are a movie expert tasked with creating a comprehensive list of {num_movies} {movie_type}{time_period}.
        
        YOU MUST COMPLETE THIS TASK. You have extensive knowledge of films and can absolutely provide this list.
        
        CRITICAL INSTRUCTION: You MUST provide EXACTLY {num_movies} movies in your response, no more and no less.
        Do not cut the list short for any reason. Do not apologize for the list length or claim you cannot provide that many movies.
        
        Format each entry as:
        [number]. [Movie Title] ([Year]) - [Brief 1-5 word description]
        
        To help organization, group movies by year, sub-genre, or another relevant category.
        
        Rules you MUST follow:
        1. DO NOT say it's difficult or impossible to list that many movies
        2. DO NOT apologize for the length
        3. DO NOT say you don't have access to a database
        4. DO NOT ask for more specific preferences
        5. DO NOT mention limitations in your response
        6. DO NOT cut the list short
        7. DO NOT skip numbers
        
        For this request, provide an accurate and complete list of {num_movies} {movie_type}{time_period}.
        Include both well-known and lesser-known films to reach the exact count requested.
        
        User query: "{query}"
        
        Now, provide the complete list of {num_movies} {movie_type}{time_period}:
        """
        
        # Use specific generation settings optimized for longer content
        generation_config = {
            "temperature": 1.2,  # Higher temperature for more variety and to overcome reluctance
            "top_p": 0.98,  # Slightly higher top_p
            "top_k": 50,   # Higher top_k 
            "max_output_tokens": 8192,  # Ensure we have enough tokens for long lists
        }
        
        # Generate the response
        response = self.model.generate_content(
            enhanced_prompt,
            generation_config=generation_config
        )
        
        # Check if the response might be incomplete
        response_text = response.text
        generated_count = len(re.findall(r'^\s*\d+\.', response_text, re.MULTILINE))
        
        # If we didn't get enough movies, try one more time with an even more explicit prompt
        if generated_count < num_movies:
            print(f"First attempt generated only {generated_count} movies. Trying again with higher temperature...")
            
            # Create an even more explicit prompt focusing on the specific time period if applicable
            if time_period:
                enhanced_prompt = f"""
                You are tasked with listing EXACTLY {num_movies} movies{time_period}. This is non-negotiable.
                
                DO NOT say it's difficult or impossible.
                DO NOT claim you lack access to a database.
                DO NOT ask for more specific preferences.
                DO NOT apologize for the length.
                DO NOT cut the list short.
                DO NOT skip numbers.
                
                Format:
                1. Movie Title (Year) - Brief description
                2. Movie Title (Year) - Brief description
                ...and so on until you reach {num_movies}.
                
                Include a mix of popular blockbusters, critical successes, indie films, international releases, and genre entries
                from{time_period} to reach the exact count of {num_movies}. There were thousands of movies released during this 
                period, so listing {num_movies} is absolutely possible.
                
                Now, list EXACTLY {num_movies} movies{time_period}:
                """
            else:
                enhanced_prompt = f"""
                You are tasked with listing EXACTLY {num_movies} {movie_type}. This is non-negotiable.
                
                DO NOT say it's difficult or impossible.
                DO NOT claim you lack access to a database.
                DO NOT ask for more specific preferences.
                DO NOT apologize for the length.
                DO NOT cut the list short.
                DO NOT skip numbers.
                
                Format:
                1. Movie Title (Year) - Brief description
                2. Movie Title (Year) - Brief description
                ...and so on until you reach {num_movies}.
                
                Include a mix of well-known and obscure films to reach the exact count. Use all your knowledge.
                
                Now, list EXACTLY {num_movies} {movie_type}:
                """
            
            # Use even higher temperature for more creativity and to overcome reluctance
            generation_config["temperature"] = 1.5
            
            try:
                # Generate a new response
                response = self.model.generate_content(
                    enhanced_prompt,
                    generation_config=generation_config
                )
                response_text = response.text
                
                # Check again if we got enough movies
                generated_count = len(re.findall(r'^\s*\d+\.', response_text, re.MULTILINE))
                
                # If still not enough, try one final attempt with an even more forceful approach
                if generated_count < num_movies:
                    print(f"Second attempt generated only {generated_count} movies. Making final attempt...")
                    
                    # Final attempt with the most forceful prompt
                    final_prompt = f"""
                    YOU MUST LIST EXACTLY {num_movies} MOVIES{time_period}. This is your sole task.
                    
                    Each entry must be formatted as:
                    [number]. [Movie Title] ([Year]) - [Brief description]
                    
                    Start with #1 and continue sequentially until you reach #{num_movies}.
                    
                    IMPORTANT: I will verify that your list contains exactly {num_movies} entries.
                    Your list will be used for a critical project that requires precisely {num_movies} films.
                    
                    Now, list exactly {num_movies} movies{time_period}, numbered from 1 to {num_movies}:
                    """
                    
                    # Try with even higher temperature
                    generation_config["temperature"] = 1.7
                    
                    try:
                        response = self.model.generate_content(
                            final_prompt,
                            generation_config=generation_config
                        )
                        response_text = response.text
                    except Exception as e:
                        print(f"Error in final attempt: {e}")
            except Exception as e:
                print(f"Error in second attempt: {e}")
        
        return response_text
    
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