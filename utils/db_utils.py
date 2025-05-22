"""
Database utilities for persisting chat history using Supabase only
Using a simplified approach that works with Python 3.13
"""
import os
import uuid
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SupabaseManager:
    """Manages database operations for chat history persistence using direct Supabase REST API"""
    
    def __init__(self):
        """Initialize the database connection"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.connected = False
        
        if not self.supabase_url or not self.supabase_key:
            print("Warning: Supabase credentials not found. Persistent storage disabled.")
            return
            
        try:
            # Test connection with a simple request
            headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(f"{self.supabase_url}/rest/v1/chat_sessions?limit=1", headers=headers)
            
            if response.status_code in (200, 201, 206):
                self.connected = True
                print("Successfully connected to Supabase")
            else:
                print(f"Failed to connect to Supabase: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error connecting to Supabase: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Supabase"""
        return self.connected
    
    def _make_request(self, method, endpoint, data=None, params=None):
        """Make a request to Supabase REST API"""
        if not self.is_connected():
            return None
            
        url = f"{self.supabase_url}/rest/v1/{endpoint}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        
        try:
            if method.lower() == 'get':
                response = requests.get(url, headers=headers, params=params)
            elif method.lower() == 'post':
                response = requests.post(url, headers=headers, json=data)
            elif method.lower() == 'put':
                response = requests.put(url, headers=headers, json=data)
            elif method.lower() == 'patch':
                response = requests.patch(url, headers=headers, json=data)
            elif method.lower() == 'delete':
                response = requests.delete(url, headers=headers, params=params)
            else:
                print(f"Unsupported method: {method}")
                return None
                
            if response.status_code in (200, 201, 204, 206):
                if response.text:
                    return response.json()
                return True
            else:
                print(f"Request failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error making request: {e}")
            return None
    
    def create_session(self, session_name: Optional[str] = None) -> str:
        """
        Create a new chat session
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session ID
        """
        if not self.is_connected():
            return str(uuid.uuid4())  # Return a temporary ID if not connected
            
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Create session metadata
        session_data = {
            "id": session_id,
            "name": session_name or f"Movie Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        result = self._make_request('post', 'chat_sessions', data=session_data)
        if result:
            return session_id
        return session_id  # Return the ID anyway for local use
    
    def get_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all available chat sessions
        
        Returns:
            List of session objects
        """
        if not self.is_connected():
            return []
            
        result = self._make_request('get', 'chat_sessions?order=updated_at.desc')
        return result or []
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session
        
        Args:
            session_id: The session ID
            
        Returns:
            List of conversation messages
        """
        if not self.is_connected():
            return []
            
        result = self._make_request('get', f'chat_messages?select=role,content&session_id=eq.{session_id}&order=created_at.asc')
        return result or []
    
    def save_message(self, session_id: str, role: str, content: str) -> bool:
        """
        Save a message to the database
        
        Args:
            session_id: The session ID
            role: Message role (user or assistant)
            content: Message content
            
        Returns:
            Success flag
        """
        if not self.is_connected():
            return False
            
        # Update session timestamp
        self._make_request('patch', f'chat_sessions?id=eq.{session_id}', 
                          data={"updated_at": datetime.now().isoformat()})
        
        # Insert message
        message_data = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.now().isoformat()
        }
        
        result = self._make_request('post', 'chat_messages', data=message_data)
        return result is not None
    
    def save_history(self, session_id: str, history: List[Dict[str, str]]) -> bool:
        """
        Save entire conversation history
        
        Args:
            session_id: The session ID
            history: List of conversation messages
            
        Returns:
            Success flag
        """
        if not self.is_connected():
            return False
            
        success = True
        for message in history:
            role = message.get("role")
            content = message.get("content")
            if role and content:
                result = self.save_message(session_id, role, content)
                success = success and result
                
        return success
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session and its messages
        
        Args:
            session_id: The session ID
            
        Returns:
            Success flag
        """
        if not self.is_connected():
            return False
            
        # Delete messages first
        self._make_request('delete', f'chat_messages?session_id=eq.{session_id}')
        
        # Delete session
        result = self._make_request('delete', f'chat_sessions?id=eq.{session_id}')
        return result is not None
    
    def get_all_session_histories(self) -> List[List[Dict[str, str]]]:
        """
        Get conversation histories for all available sessions
        
        Returns:
            List of conversation histories, one for each session
        """
        if not self.is_connected():
            return []
            
        # Get all session IDs
        sessions = self.get_sessions()
        if not sessions:
            return []
            
        # Retrieve history for each session
        all_histories = []
        for session in sessions:
            session_id = session.get("id")
            if session_id:
                history = self.get_session_history(session_id)
                if history:
                    all_histories.append(history)
                    
        return all_histories

# Create the main db_manager instance - only Supabase, no local fallback
db_manager = SupabaseManager()
if not db_manager.is_connected():
    print("Warning: Supabase connection failed. No local fallback will be used.")
    print("Please check your Supabase credentials in the .env file.")
    print("Your data will only be stored in memory for this session.")