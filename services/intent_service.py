"""
Service for recognizing user intents using a hybrid approach
"""
import re
from typing import Dict, Any, List, Tuple, Optional
import os
import json
from collections import defaultdict

class IntentService:
    """
    Service for recognizing user intents using a hybrid approach:
    1. Rule-based pattern matching
    2. Keyword-based intent classification
    3. LLM-based intent recognition (through GeminiService)
    """
    
    def __init__(self, gemini_service=None):
        """Initialize the Intent Service with optional gemini service integration"""
        self.gemini_service = gemini_service
        
        # Define common movie-related intents with patterns and keywords
        self.intent_patterns = {
            "recommendation": [
                r"(?:recommend|suggest)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:what|any)(?:\s+\w+){0,3}(?:\s+movie)(?:\s+\w+){0,3}(?:watch|see)",
                r"(?:looking\s+for)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:can\s+you\s+suggest)(?:\s+\w+){0,5}(?:\s+movie)",
            ],
            "information": [
                r"(?:who|what|when|where|why|how)(?:\s+\w+){0,10}(?:\s+movie\s+)",
                r"(?:tell\s+me\s+about)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:information\s+on)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:details\s+about)(?:\s+\w+){0,5}",
            ],
            "comparison": [
                r"(?:compare|versus|vs|or|better)(?:\s+\w+){0,10}(?:\s+movie)",
                r"(?:which\s+is\s+better)(?:\s+\w+){0,10}",
                r"(?:difference\s+between)(?:\s+\w+){0,10}",
            ],
            "rating": [
                r"(?:rating|score|review)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:how\s+good\s+is)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:is)(?:\s+\w+){0,5}(?:\s+movie)(?:\s+\w+){0,3}(?:good|great|worth)",
            ],
            "genre": [
                r"(?:action|comedy|drama|horror|thriller|sci-fi|romance|documentary|animated)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:movie)(?:\s+\w+){0,3}(?:\s+genre)",
                r"(?:like\s+a)(?:\s+\w+){0,3}(?:action|comedy|drama|horror|thriller|sci-fi|romance)(?:\s+\w+){0,3}(?:\s+movie)",
            ],
            "actor": [
                r"(?:movie)(?:\s+\w+){0,5}(?:\s+star|actor|actress)",
                r"(?:actor|actress)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:who\s+(?:stars|acted|plays))(?:\s+\w+){0,10}",
            ],
            "director": [
                r"(?:directed\s+by|director\s+of)(?:\s+\w+){0,5}",
                r"(?:who\s+directed)(?:\s+\w+){0,5}",
                r"(?:movie)(?:\s+\w+){0,5}(?:\s+directed\s+by)",
            ],
            "release": [
                r"(?:when|what\s+year|release\s+date)(?:\s+\w+){0,5}(?:\s+release|come\s+out)",
                r"(?:how\s+old\s+is)(?:\s+\w+){0,5}(?:\s+movie)",
                r"(?:released\s+in)(?:\s+\w+){0,5}",
            ],
            "plot": [
                r"(?:what\s+(?:happens|is\s+it\s+about)|plot|story|synopsis)(?:\s+\w+){0,5}",
                r"(?:tell\s+me\s+the\s+plot)(?:\s+\w+){0,5}",
                r"(?:what\s+is)(?:\s+\w+){0,5}(?:\s+about)",
            ],
            "similar": [
                r"(?:similar\s+to|like|movies?\s+like)(?:\s+\w+){0,5}",
                r"(?:more\s+(?:movies?|films?)\s+like)(?:\s+\w+){0,5}",
                r"(?:recommend)(?:\s+\w+){0,3}(?:\s+like)(?:\s+\w+){0,5}",
            ],
            "preference_update": [
                r"(?:i\s+(?:like|love|enjoy|prefer))(?:\s+\w+){0,10}",
                r"(?:i\s+(?:hate|dislike|don't\s+like))(?:\s+\w+){0,10}",
                r"(?:my\s+favorite)(?:\s+\w+){0,10}",
            ],
            "greeting": [
                r"^(?:hi|hello|hey|greetings|howdy)(?:\s+\w+){0,3}$",
                r"^(?:good\s+(?:morning|afternoon|evening))(?:\s+\w+){0,3}$",
            ],
            "thanks": [
                r"^(?:thanks|thank\s+you|thx|ty)(?:\s+\w+){0,5}$",
                r"^(?:appreciate\s+(?:it|that))(?:\s+\w+){0,5}$",
            ],
            "help": [
                r"(?:help|assist|guide)(?:\s+\w+){0,5}",
                r"(?:how\s+(?:do|can|should)\s+(?:i|you))(?:\s+\w+){0,5}",
                r"(?:what\s+can\s+you\s+do)(?:\s+\w+){0,5}",
            ],
            "list_reference": [
                r"(?:the|that)\s+(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)\s+(?:movie|film|one)",
                r"(?:movie|film)\s+(?:number)?\s+(\d+)",
                r"(?:number|#)\s+(\d+)",
                r"(\d+)(?:st|nd|rd|th)\s+(?:movie|film|one)",
                r"(\d+)(?:st|nd|rd|th)",
                r"(?:give|tell|detail|describe|explain)(?:\s+\w+){0,5}(?:plot|story|synopsis)(?:\s+\w+){0,5}(?:of)?\s+(?:the)?\s*(\d+)(?:st|nd|rd|th)?(?:\s+movie|film)?",
                r"(?:give|tell|detail|describe|explain)(?:\s+\w+){0,5}(?:plot|story|synopsis)(?:\s+\w+){0,5}(?:of)?\s+(?:the)?\s*(\d+)(?:st|nd|rd|th)?",
            ],
            "continue_topic": [
                r"^(?:more|continue|go on|tell me more)$",
                r"^(?:and|what else|continue|furthermore)(?:\s+\w+){0,3}$",
                r"^(?:can you elaborate)(?:\s+\w+){0,5}$",
            ],
        }
        
        # Define keywords for each intent
        self.intent_keywords = {
            "recommendation": [
                "recommend", "suggest", "suggestion", "recommendation", "looking for", 
                "find me", "can you find", "what should", "movie to watch", "show me"
            ],
            "information": [
                "information", "details", "tell me about", "what is", "facts", "trivia",
                "who is", "when was", "where was", "why did"
            ],
            "comparison": [
                "compare", "versus", "vs", "or", "better", "which one", "difference between",
                "similarities", "pros and cons", "which is better"
            ],
            "rating": [
                "rating", "score", "review", "critics", "how good", "worth watching",
                "metacritic", "rotten tomatoes", "imdb", "stars"
            ],
            "genre": [
                "action", "comedy", "drama", "horror", "thriller", "sci-fi", "science fiction",
                "romance", "documentary", "animated", "genre", "category", "type of movie"
            ],
            "actor": [
                "actor", "actress", "star", "cast", "who plays", "who starred", "role",
                "performer", "character", "acted"
            ],
            "director": [
                "director", "directed by", "filmmaker", "who directed", "made by",
                "creator", "directed", "film by"
            ],
            "release": [
                "release", "came out", "released", "premiere", "debut", "when did",
                "year", "date", "how old", "when was it"
            ],
            "plot": [
                "plot", "story", "synopsis", "what happens", "about", "storyline",
                "narrative", "what is it about", "summary", "premise"
            ],
            "similar": [
                "similar", "like", "movies like", "comparable", "in the same vein",
                "in the style of", "reminds me of", "same genre as", "more like"
            ],
            "preference_update": [
                "I like", "I love", "I enjoy", "I prefer", "I hate", "I dislike",
                "don't like", "my favorite", "I'm into", "I'm a fan of"
            ],
            "greeting": [
                "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
                "greetings", "howdy", "what's up"
            ],
            "thanks": [
                "thanks", "thank you", "appreciate it", "grateful", "thx", "ty",
                "thank", "appreciate", "appreciated"
            ],
            "help": [
                "help", "assist", "guide", "how do I", "what can you do", "instructions",
                "capabilities", "features", "functions", "assistance"
            ],
            "list_reference": [
                "the first", "the second", "the third", "the fourth", "the fifth",
                "the sixth", "the seventh", "the eighth", "the ninth", "the tenth",
                "number 1", "number 2", "number 3", "number 4", "number 5",
                "number 6", "number 7", "number 8", "number 9", "number 10",
                "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10",
                "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
                "first movie", "second movie", "third movie", "fourth movie", "fifth movie",
                "movie number", "film number", "20th", "30th", "40th", "50th", "60th", "70th", "80th", "90th", "100th",
                "number 20", "number 30", "number 40", "number 50", "number 60", "number 70", "number 80", "number 90", "number 100"
            ],
            "continue_topic": [
                "more", "continue", "go on", "tell me more", "and", "what else",
                "furthermore", "elaborate", "can you elaborate", "additional information"
            ],
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.intent_patterns.items()
        }
        
        # Load any additional custom patterns if available
        self.custom_patterns_path = os.path.join(os.path.dirname(__file__), "..", "data", "custom_intent_patterns.json")
        self.load_custom_patterns()
        
        # Entity extraction patterns
        self.entity_patterns = {
            "movie_title": [
                r'"([^"]+)"',  # Anything in quotes
                r"'([^']+)'",  # Anything in single quotes
                r"(?:the\s+movie|film|picture)\s+(?:called|titled|named)?\s+([A-Z][a-zA-Z0-9\s:&'-]+)",  # The movie called X
                r"(?:watched|seen|liked|loved|hated)\s+(?:the\s+movie|film)?\s+([A-Z][a-zA-Z0-9\s:&'-]+)",  # Watched X
            ],
            "person_name": [
                r"(?:actor|actress|director|star)\s+([A-Z][a-zA-Z\s-]+)",  # Actor X
                r"(?:by|with)\s+([A-Z][a-zA-Z\s-]+)",  # By/With X
                r"(?:like|enjoy|fan\s+of)\s+([A-Z][a-zA-Z\s-]+)(?:'s)?",  # Like X
            ],
            "genre": [
                r"(?:genre|category|type)\s+(?:of|like)?\s+([a-zA-Z\s-]+)",  # Genre of X
                r"([a-zA-Z-]+)\s+(?:movies|films)",  # X movies
            ],
            "year": [
                r"(?:in|from|during|year)\s+(\d{4})",  # In 2023
                r"(\d{4})(?:'s)?",  # 2023, 2023's
            ],
            "rating": [
                r"(?:rated|rating|score)\s+(?:of)?\s+(\d+(?:\.\d+)?)",  # Rating of 8.5
                r"(\d+(?:\.\d+)?)\s+(?:out\s+of|\/)\s+(\d+)",  # 8.5 out of 10
            ],
            "list_index": [
                r"(?:number|#)?\s*(\d+)(?:st|nd|rd|th)?(?:\s+movie|film|one)?",  # #5, number 5, 5th movie
                r"(?:the\s+)?(\d+)(?:st|nd|rd|th)(?:\s+movie|film|one)?",  # the 5th, 5th movie
                r"(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)(?:\s+movie|film|one)?",  # the fifth, fifth movie
                r"(?:give|tell|detail|describe|explain)(?:\s+\w+){0,5}(?:plot|story|synopsis)(?:\s+\w+){0,5}(?:of)?\s+(?:the)?\s*(\d+)(?:st|nd|rd|th)?(?:\s+movie|film)?",  # give the plot of the 5th movie
                r"(?:give|tell|detail|describe|explain)(?:\s+\w+){0,5}(?:the)?\s*(\d+)(?:st|nd|rd|th)?(?:\s+movie|film)",  # tell me about the 5th movie
            ],
        }
        
        # Number word to integer mapping
        self.number_word_map = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, 
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
            "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20
        }
        
        # Compile entity patterns
        self.compiled_entity_patterns = {
            entity: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for entity, patterns in self.entity_patterns.items()
        }
        
        # Initialize conversational context tracking
        self.conversation_context = ConversationContext()
    
    def load_custom_patterns(self):
        """Load custom intent patterns from a JSON file if available"""
        try:
            if os.path.exists(self.custom_patterns_path):
                with open(self.custom_patterns_path, 'r') as f:
                    custom_patterns = json.load(f)
                
                # Add custom patterns to existing ones
                for intent, patterns in custom_patterns.items():
                    if intent in self.intent_patterns:
                        self.intent_patterns[intent].extend(patterns)
                    else:
                        self.intent_patterns[intent] = patterns
                    
                    # Recompile patterns for this intent
                    self.compiled_patterns[intent] = [
                        re.compile(pattern, re.IGNORECASE) 
                        for pattern in self.intent_patterns[intent]
                    ]
        except Exception as e:
            print(f"Warning: Failed to load custom intent patterns: {e}")
    
    def recognize_intent(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Recognize the intent of a user query using a hybrid approach
        
        Args:
            query: The user's query text
            conversation_history: Optional conversation history for context
            
        Returns:
            Dictionary with recognized intent, confidence, and entities
        """
        # Update conversation context
        if conversation_history:
            self.conversation_context.update_from_history(conversation_history)
        
        # 1. Rule-based pattern matching
        rule_based_intents = self._match_patterns(query)
        
        # 2. Keyword-based intent classification
        keyword_intents = self._match_keywords(query)
        
        # 3. Entity extraction
        entities = self._extract_entities(query)
        
        # 4. Process list references if present
        result = self._process_list_references(query, entities)
        if result and result.get("resolved"):
            # If we resolved a list reference, update entities and override intent
            entities.update(result.get("entities", {}))
            resolved_intent = {
                result.get("intent", "information"): 1.0
            }
            return {
                "intent": result.get("intent", "information"),
                "confidence": 1.0,
                "entities": entities,
                "all_intents": resolved_intent,
                "resolved_reference": True,
                "reference_data": result.get("reference_data")
            }
        
        # 5. LLM-based intent recognition (if gemini_service is available)
        llm_intent = None
        llm_confidence = 0
        
        if self.gemini_service:
            llm_result = self._get_llm_intent(query, conversation_history)
            llm_intent = llm_result.get("intent")
            llm_confidence = llm_result.get("confidence", 0)
            
            # Merge LLM-detected entities with our rule-based ones
            llm_entities = llm_result.get("entities", {})
            for entity_type, entity_values in llm_entities.items():
                if entity_type in entities:
                    # Add unique values only
                    entities[entity_type] = list(set(entities[entity_type] + entity_values))
                else:
                    entities[entity_type] = entity_values
        
        # Combine results from all approaches (hybrid approach)
        intent_scores = defaultdict(float)
        
        # Add rule-based results (highest weight)
        for intent, score in rule_based_intents.items():
            intent_scores[intent] += score * 0.5  # 50% weight
            
        # Add keyword-based results (medium weight)
        for intent, score in keyword_intents.items():
            intent_scores[intent] += score * 0.3  # 30% weight
            
        # Add LLM-based results (lowest weight, but can be decisive for ambiguous cases)
        if llm_intent:
            intent_scores[llm_intent] += llm_confidence * 0.2  # 20% weight
        
        # Get the intent with the highest combined score
        if intent_scores:
            top_intent = max(intent_scores.items(), key=lambda x: x[1])
            intent_name = top_intent[0]
            confidence = min(top_intent[1], 1.0)  # Cap confidence at 1.0
        else:
            # Default to "information" if no intent was detected
            intent_name = "information"
            confidence = 0.3
        
        return {
            "intent": intent_name,
            "confidence": confidence,
            "entities": entities,
            "all_intents": dict(intent_scores)
        }
    
    def _match_patterns(self, query: str) -> Dict[str, float]:
        """
        Match query against regex patterns for intent detection
        
        Args:
            query: The user's query text
            
        Returns:
            Dictionary of intent names and confidence scores
        """
        results = {}
        
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    # If we found a match, add 1.0 to this intent's score
                    # (we'll normalize later if needed)
                    results[intent] = results.get(intent, 0) + 1.0
        
        # Normalize scores if we have any matches
        if results:
            max_score = max(results.values())
            for intent in results:
                results[intent] /= max_score
        
        return results
    
    def _match_keywords(self, query: str) -> Dict[str, float]:
        """
        Match query against keywords for intent detection
        
        Args:
            query: The user's query text
            
        Returns:
            Dictionary of intent names and confidence scores
        """
        query_lower = query.lower()
        results = {}
        
        for intent, keywords in self.intent_keywords.items():
            matches = 0
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    matches += 1
            
            if matches > 0:
                # Calculate score based on number of keyword matches
                results[intent] = min(matches / len(keywords) * 2, 1.0)
        
        return results
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from the query using regex patterns
        
        Args:
            query: The user's query text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = defaultdict(list)
        
        for entity_type, patterns in self.compiled_entity_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(query)
                if matches:
                    # Handle different match structures (tuple vs string)
                    for match in matches:
                        if isinstance(match, tuple):
                            for group in match:
                                if group and group not in entities[entity_type]:
                                    entities[entity_type].append(group)
                        elif match and match not in entities[entity_type]:
                            entities[entity_type].append(match)
        
        # For list_index entities, convert word numbers to integers
        if "list_index" in entities:
            numeric_indices = []
            for idx in entities["list_index"]:
                try:
                    # Try to convert to integer directly
                    numeric_indices.append(int(idx))
                except ValueError:
                    # Check if it's a word number
                    if idx.lower() in self.number_word_map:
                        numeric_indices.append(self.number_word_map[idx.lower()])
            
            if numeric_indices:
                entities["list_index"] = numeric_indices
        
        # Special case for high number references (e.g., "90th movie")
        # Try a more direct extraction for these cases
        if "list_index" not in entities or not entities["list_index"]:
            direct_index_pattern = r"(\d+)(?:st|nd|rd|th)?\s*(?:movie|film|one)?$"
            match = re.search(direct_index_pattern, query)
            if match:
                try:
                    index = int(match.group(1))
                    entities["list_index"] = [index]
                except ValueError:
                    pass
            
            # Also check for "90th", "70th", etc. specifically
            direct_index_pattern = r"(?:the\s+)?(\d+)(?:st|nd|rd|th)"
            match = re.search(direct_index_pattern, query)
            if match:
                try:
                    index = int(match.group(1))
                    entities["list_index"] = [index]
                except ValueError:
                    pass
        
        return dict(entities)
    
    def _extract_movie_list(self, text: str) -> List[str]:
        """Extract a list of movies from text"""
        # Find all numbered list items
        list_items = []
        
        # Try to match numbered items like: "90 Memento (2000) - Psychological thriller"
        numbered_pattern = r'(?:^|\n)\s*(\d+)\s+(.*?)(?=\n\s*\d+\s+|$)'
        items = re.findall(numbered_pattern, text, re.DOTALL)
        
        if items:
            # Create a dictionary to store items by their number
            numbered_items = {}
            for item_num, item_text in items:
                try:
                    num = int(item_num)
                    # Clean up the item text
                    cleaned = re.sub(r'\s+', ' ', item_text).strip()
                    if cleaned:
                        numbered_items[num] = cleaned
                except (ValueError, TypeError):
                    continue
            
            # Convert to a list, preserving the exact numbering
            max_num = max(numbered_items.keys()) if numbered_items else 0
            list_items = [""] * max_num  # Initialize with empty strings
            
            # Fill in the items at their correct positions
            for num, item in numbered_items.items():
                if 1 <= num <= max_num:  # Ensure valid index
                    list_items[num-1] = item  # Adjust for 0-based indexing
            
            # Remove any empty strings
            list_items = [item for item in list_items if item]
        
        # If we didn't find items with the expected pattern, fall back to a simpler approach
        if not list_items:
            # Original extraction logic as fallback
            lines = text.split('\n')
            numbered_items = {}
            for line in lines:
                # Look for lines that start with a number
                match = re.match(r'\s*(\d+)[\.\):]?\s+(.*)', line)
                if match and match.group(2).strip():
                    try:
                        item_num = int(match.group(1))
                        item_text = match.group(2).strip()
                        numbered_items[item_num] = item_text
                    except (ValueError, TypeError):
                        continue
            
            # Convert to a list, preserving the exact numbering
            if numbered_items:
                max_num = max(numbered_items.keys())
                list_items = [""] * max_num
                for num, item in numbered_items.items():
                    if 1 <= num <= max_num:
                        list_items[num-1] = item
                
                # Remove any empty strings
                list_items = [item for item in list_items if item]
        
        # Debug logging
        if list_items:
            print(f"Extracted {len(list_items)} movie list items")
            if len(list_items) >= 90:
                print(f"Item #90: {list_items[89]}")
            if len(list_items) >= 93:
                print(f"Item #93: {list_items[92]}")
        
        return list_items

    def _process_list_references(self, query: str, entities: Dict[str, List]) -> Optional[Dict[str, Any]]:
        """
        Process list references in the query (e.g., "the 3rd movie", "number 5")
        
        Args:
            query: The user's query
            entities: Extracted entities
            
        Returns:
            Dictionary with resolved reference information or None
        """
        # Check if there's a list index in the extracted entities
        if "list_index" not in entities or not entities["list_index"]:
            # Try to extract numbers directly from the query as a last resort
            direct_number_match = re.search(r"(\d+)(?:st|nd|rd|th)?", query)
            if direct_number_match:
                try:
                    index = int(direct_number_match.group(1))
                    entities["list_index"] = [index]
                except (ValueError, TypeError):
                    return None
            else:
                return None
        
        # Get the referenced index
        index = entities["list_index"][0]
        if not isinstance(index, int):
            try:
                index = int(index)
            except (ValueError, TypeError):
                return None
        
        # Log the requested index
        print(f"Processing list reference for index: {index}")
        
        # Get the most recent list from conversation context
        recent_list = self.conversation_context.get_most_recent_list()
        if not recent_list:
            print("No recent list found in conversation context")
            return None
        
        print(f"Found recent list with {len(recent_list)} items")
        
        # Adjust for zero-indexing
        adjusted_index = index - 1
        
        # Check if the index is valid
        if adjusted_index < 0 or adjusted_index >= len(recent_list):
            print(f"Index {index} is out of bounds for list of length {len(recent_list)}")
            return None
        
        # Get the referenced item
        referenced_item = recent_list[adjusted_index]
        print(f"Referenced item at index {index}: {referenced_item}")
        
        # Extract movie information from the referenced item
        movie_title = None
        year = None
        
        # Try different parsing strategies
        if isinstance(referenced_item, str):
            # Strategy 1: Look for "NUMBER. Movie Title (YEAR) - Description" or "Movie Title (YEAR) - Description"
            title_match = re.search(r'^(?:\d+\.?\s+)?([^(]+)\s*\((\d{4})\)\s*-\s*', referenced_item)
            if title_match:
                movie_title = title_match.group(1).strip()
                year = title_match.group(2)
                print(f"Strategy 1: Extracted movie '{movie_title}' ({year})")
            
            # Strategy 2: Just extract the movie name if there's a number prefix
            if not movie_title:
                title_match = re.search(r'^(?:\d+\.?\s+)?([^-]+)', referenced_item)
                if title_match:
                    movie_title = title_match.group(1).strip()
                    print(f"Strategy 2: Extracted movie '{movie_title}'")
            
            # Strategy 3: Extract title with year if present anywhere
            if not movie_title:
                title_match = re.search(r'([^-:]+)\s*\((\d{4})\)', referenced_item)
                if title_match:
                    movie_title = title_match.group(1).strip()
                    year = title_match.group(2)
                    print(f"Strategy 3: Extracted movie '{movie_title}' ({year})")
            
            # Strategy 4: Just use the raw text as a last resort
            if not movie_title:
                # Remove any numbering at the beginning
                movie_title = re.sub(r'^\s*\d+\.?\s*', '', referenced_item).strip()
                print(f"Strategy 4: Using raw text as movie title: '{movie_title}'")
        elif isinstance(referenced_item, dict) and "title" in referenced_item:
            movie_title = referenced_item["title"]
            year = referenced_item.get("year")
            print(f"Extracted movie from dictionary: '{movie_title}'")
        
        if not movie_title:
            print("Failed to extract movie title from referenced item")
            # Use the whole item as a fallback
            movie_title = str(referenced_item).strip() if referenced_item else "unknown movie"
        
        # Determine the intent based on the query
        intent = "information"
        if "plot" in query.lower() or "story" in query.lower() or "synopsis" in query.lower():
            intent = "plot"
        elif "actor" in query.lower() or "cast" in query.lower() or "star" in query.lower():
            intent = "actor"
        elif "director" in query.lower() or "directed" in query.lower():
            intent = "director"
        elif "rating" in query.lower() or "score" in query.lower() or "review" in query.lower():
            intent = "rating"
        elif "release" in query.lower() or "year" in query.lower() or "when" in query.lower():
            intent = "release"
        
        return {
            "resolved": True,
            "intent": intent,
            "entities": {
                "movie_title": [movie_title]
            },
            "reference_data": {
                "index": index,  # Keep as 1-based for human readability
                "item": referenced_item,
                "year": year
            }
        }
    
    def _extract_movie_list(self, text: str) -> List[str]:
        """Extract a list of movies from text"""
        # Find all numbered list items
        list_items = []
        
        # Try to match numbered items like: "90 Memento (2000) - Psychological thriller"
        numbered_pattern = r'(?:^|\n)\s*(\d+)\s+(.*?)(?=\n\s*\d+\s+|$)'
        items = re.findall(numbered_pattern, text, re.DOTALL)
        
        if items:
            # Create a dictionary to store items by their number
            numbered_items = {}
            for item_num, item_text in items:
                try:
                    num = int(item_num)
                    # Clean up the item text
                    cleaned = re.sub(r'\s+', ' ', item_text).strip()
                    if cleaned:
                        numbered_items[num] = cleaned
                except (ValueError, TypeError):
                    continue
            
            # Convert to a list, preserving the exact numbering
            max_num = max(numbered_items.keys()) if numbered_items else 0
            list_items = [""] * max_num  # Initialize with empty strings
            
            # Fill in the items at their correct positions
            for num, item in numbered_items.items():
                if 1 <= num <= max_num:  # Ensure valid index
                    list_items[num-1] = item  # Adjust for 0-based indexing
            
            # Remove any empty strings
            list_items = [item for item in list_items if item]
        
        # If we didn't find items with the expected pattern, fall back to a simpler approach
        if not list_items:
            # Original extraction logic as fallback
            lines = text.split('\n')
            numbered_items = {}
            for line in lines:
                # Look for lines that start with a number
                match = re.match(r'\s*(\d+)[\.\):]?\s+(.*)', line)
                if match and match.group(2).strip():
                    try:
                        item_num = int(match.group(1))
                        item_text = match.group(2).strip()
                        numbered_items[item_num] = item_text
                    except (ValueError, TypeError):
                        continue
            
            # Convert to a list, preserving the exact numbering
            if numbered_items:
                max_num = max(numbered_items.keys())
                list_items = [""] * max_num
                for num, item in numbered_items.items():
                    if 1 <= num <= max_num:
                        list_items[num-1] = item
                
                # Remove any empty strings
                list_items = [item for item in list_items if item]
        
        # Debug logging
        if list_items:
            print(f"Extracted {len(list_items)} movie list items")
            if len(list_items) >= 90:
                print(f"Item #90: {list_items[89]}")
            if len(list_items) >= 93:
                print(f"Item #93: {list_items[92]}")
        
        return list_items
    
class ConversationContext:
    """Class to maintain conversation context across turns"""
    
    def __init__(self):
        """Initialize the conversation context"""
        self.recent_lists = []  # Most recent lists (e.g., movie recommendations)
        self.recent_movies = []  # Recently mentioned movies
        self.recent_actors = []  # Recently mentioned actors
        self.recent_directors = []  # Recently mentioned directors
        self.recent_genres = []  # Recently mentioned genres
        self.current_topic = None  # Current topic of conversation
        
        # Maximum number of items to keep in each list
        self.max_items = 10
        
        # Maximum number of lists to keep
        self.max_lists = 5
        
        # Lock the current list to prevent modification after creation
        self.current_list_locked = False
        
        # Debugging information
        self.debug_info = []
    
    def update_from_history(self, history: List[Dict[str, str]]):
        """
        Update context from conversation history
        
        Args:
            history: List of conversation messages
        """
        # Process the most recent messages (last 10)
        recent_messages = history[-10:] if len(history) > 10 else history
        
        for message in recent_messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "assistant" and content:
                # Check if the assistant's message contains a list
                if self._contains_movie_list(content):
                    movie_list = self._extract_movie_list(content)
                    if movie_list:
                        self.add_list(movie_list)
    
    def add_list(self, items: List[str]):
        """
        Add a list of items to the context
        
        Args:
            items: List of items (e.g., movie titles)
        """
        # Store debugging info about the list
        list_debug = {
            "action": "add_list",
            "item_count": len(items),
            "first_item": items[0] if items else None,
            "last_item": items[-1] if items else None,
            "is_locked": self.current_list_locked
        }
        self.debug_info.append(list_debug)
        
        # If the list is locked, don't replace it - this prevents list drift
        if self.current_list_locked and self.recent_lists:
            print("List is locked - not replacing current list")
            return
        
        # Add to the beginning (most recent)
        self.recent_lists.insert(0, items.copy())  # Make a copy to prevent reference issues
        
        # Lock the list to prevent modifications that could cause index drift
        self.current_list_locked = True
        
        # Trim to max lists
        if len(self.recent_lists) > self.max_lists:
            self.recent_lists = self.recent_lists[:self.max_lists]
        
        # Print debug info
        print(f"Added list to context with {len(items)} items")
        if items:
            print(f"First item: {items[0]}")
            if len(items) > 1:
                print(f"Last item: {items[-1]}")
    
    def add_movie(self, movie: str):
        """
        Add a movie to recent movies
        
        Args:
            movie: Movie title
        """
        if movie not in self.recent_movies:
            self.recent_movies.insert(0, movie)
            
            # Trim to max items
            if len(self.recent_movies) > self.max_items:
                self.recent_movies = self.recent_movies[:self.max_items]
    
    def get_most_recent_list(self) -> Optional[List[str]]:
        """
        Get the most recently added list
        
        Returns:
            The most recent list or None if no lists exist
        """
        if self.recent_lists:
            list_items = self.recent_lists[0]
            print(f"Retrieved most recent list with {len(list_items)} items")
            print(f"First item: {list_items[0] if list_items else 'None'}")
            print(f"Last item: {list_items[-1] if list_items else 'None'}")
            
            # Store debug info about list retrieval
            retrieval_debug = {
                "action": "get_list",
                "item_count": len(list_items),
                "first_item": list_items[0] if list_items else None,
                "last_item": list_items[-1] if list_items else None
            }
            self.debug_info.append(retrieval_debug)
            
            return list_items
        return None
    
    def get_recent_movies(self) -> List[str]:
        """
        Get recently mentioned movies
        
        Returns:
            List of recent movie titles
        """
        return self.recent_movies
    
    def set_current_topic(self, topic: str):
        """
        Set the current topic of conversation
        
        Args:
            topic: The current topic
        """
        self.current_topic = topic
    
    def get_current_topic(self) -> Optional[str]:
        """
        Get the current topic of conversation
        
        Returns:
            Current topic or None if not set
        """
        return self.current_topic
    
    def _contains_movie_list(self, text: str) -> bool:
        """Check if text contains a numbered list of movies"""
        # Look for numbered list patterns (1., 2., #1, etc.)
        numbered_lines = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|#\d+|\[\d+\])\s*[A-Z]', text)
        return len(numbered_lines) >= 5  # At least 5 numbered items to consider it a list
    
    def _extract_movie_list(self, text: str) -> List[str]:
        """Extract a list of movies from text"""
        # Find all numbered list items
        list_items = []
        
        # Try to match numbered items like: "90 Memento (2000) - Psychological thriller"
        numbered_pattern = r'(?:^|\n)\s*(\d+)\s+(.*?)(?=\n\s*\d+\s+|$)'
        items = re.findall(numbered_pattern, text, re.DOTALL)
        
        if items:
            # Create a dictionary to store items by their number
            numbered_items = {}
            for item_num, item_text in items:
                try:
                    num = int(item_num)
                    # Clean up the item text
                    cleaned = re.sub(r'\s+', ' ', item_text).strip()
                    if cleaned:
                        numbered_items[num] = cleaned
                except (ValueError, TypeError):
                    continue
            
            # Convert to a list, preserving the exact numbering
            max_num = max(numbered_items.keys()) if numbered_items else 0
            list_items = [""] * max_num  # Initialize with empty strings
            
            # Fill in the items at their correct positions
            for num, item in numbered_items.items():
                if 1 <= num <= max_num:  # Ensure valid index
                    list_items[num-1] = item  # Adjust for 0-based indexing
            
            # Remove any empty strings
            list_items = [item for item in list_items if item]
        
        # If we didn't find items with the expected pattern, fall back to a simpler approach
        if not list_items:
            # Original extraction logic as fallback
            lines = text.split('\n')
            numbered_items = {}
            for line in lines:
                # Look for lines that start with a number
                match = re.match(r'\s*(\d+)[\.\):]?\s+(.*)', line)
                if match and match.group(2).strip():
                    try:
                        item_num = int(match.group(1))
                        item_text = match.group(2).strip()
                        numbered_items[item_num] = item_text
                    except (ValueError, TypeError):
                        continue
            
            # Convert to a list, preserving the exact numbering
            if numbered_items:
                max_num = max(numbered_items.keys())
                list_items = [""] * max_num
                for num, item in numbered_items.items():
                    if 1 <= num <= max_num:
                        list_items[num-1] = item
                
                # Remove any empty strings
                list_items = [item for item in list_items if item]
        
        # Debug logging
        if list_items:
            print(f"Extracted {len(list_items)} movie list items")
            if len(list_items) >= 90:
                print(f"Item #90: {list_items[89]}")
            if len(list_items) >= 93:
                print(f"Item #93: {list_items[92]}")
        
        return list_items
    
    def reset_for_new_list(self):
        """Reset the locked state to allow a new list to be stored"""
        self.current_list_locked = False
        print("Reset list lock to allow new list generation")