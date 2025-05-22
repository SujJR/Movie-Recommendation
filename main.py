#!/usr/bin/env python3
"""
Movie Recommendation Chatbot
A context-based chatbot for movie recommendations using Google's Gemini LLM.
With persistent storage of chat history using Supabase only.
"""
import os
import argparse
import uuid
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

# Load environment variables
load_dotenv()

console = Console()

# Import database manager
try:
    from utils.db_utils import db_manager
except ImportError as e:
    console.print(f"[bold red]Error:[/bold red] {e}")
    console.print("Please install required packages with 'pip install -r requirements.txt'")
    exit(1)

def run_terminal_interface(api_key=None, session_id=None, session_name=None):
    """Run the chatbot in the terminal"""
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    elif "GEMINI_API_KEY" not in os.environ:
        api_key = input("Please enter your Gemini API key: ")
        os.environ["GEMINI_API_KEY"] = api_key
    
    from services.gemini_service import GeminiService
    gemini_service = GeminiService()
    
    console.print(Panel.fit("[bold green]Movie Recommendation Chatbot[/bold green]", 
                           title="Welcome", subtitle="Personalized Movie Recommendations"))
    console.print("Type 'help' to see available commands and example queries")
    
    # Get or create session
    if session_id:
        # Load specific session from database
        history = db_manager.get_session_history(session_id)
        if history:
            console.print(f"[green]Loaded session with {len(history)} messages[/green]")
        else:
            console.print("[yellow]No messages found for this session or session doesn't exist.[/yellow]")
            session_id = db_manager.create_session(session_name)
            console.print(f"[green]Created new session (ID: {session_id[:8]}...)[/green]")
            history = []
    else:
        # Create new session
        session_id = db_manager.create_session(session_name)
        console.print(f"[green]Created new session (ID: {session_id[:8]}...)[/green]")
        history = []
    
    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ")
        
        # Handle commands
        if user_input.lower() in ('exit', 'quit'):
            console.print("[bold green]Goodbye! Hope you found some great movies to watch![/bold green]")
            console.print(f"[green]Your session ID is: {session_id}[/green]")
            console.print("[green]You can continue this conversation later using:[/green]")
            console.print(f"[green]python main.py --session {session_id}[/green]")
            break
        
        elif user_input.lower() == 'help':
            show_help()
            continue
            
        elif user_input.lower() == 'history':
            display_history(history)
            continue
            
        elif user_input.lower() == 'sessions':
            sessions = db_manager.get_sessions()
            display_sessions(sessions)
            continue
            
        elif user_input.lower() == 'load all':
                try:
                    all_histories = db_manager.get_all_session_histories()
                    if all_histories:
                        # Reset chat to ensure clean state
                        gemini_service.chat_initialized = False
                        
                        # Load the consolidated history into the service
                        consolidated_history = gemini_service.load_all_session_histories(all_histories)
                        
                        # Start with this consolidated history
                        history = consolidated_history
                        
                        console.print(f"[green]Loaded memories from {len(all_histories)} sessions[/green]")
                        console.print("[green]The AI now has access to your preferences from all previous conversations![/green]")
                    else:
                        console.print("[yellow]No previous session histories found.[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error loading all session histories: {e}[/red]")
                continue
                
        elif user_input.lower().startswith('load ') and not user_input.lower() == 'load all':
            try:
                new_session_id = user_input.split(' ', 1)[1].strip()
                new_history = db_manager.get_session_history(new_session_id)
                if new_history:
                    session_id = new_session_id
                    history = new_history
                    gemini_service.reset_chat()  # Reset chat to load history
                    console.print(f"[green]Loaded session with {len(history)} messages[/green]")
                else:
                    console.print("[yellow]No messages found for this session or session doesn't exist.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error loading session: {e}[/red]")
            continue
            
        elif user_input.lower().startswith('rename '):
            try:
                new_name = user_input.split(' ', 1)[1].strip()
                # Update session name in database
                if db_manager.is_connected():
                    db_manager.client.table("chat_sessions").update({"name": new_name}).eq("id", session_id).execute()
                    console.print(f"[green]Session renamed to '{new_name}'[/green]")
                else:
                    console.print("[yellow]Database not connected. Session not renamed.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error renaming session: {e}[/red]")
            continue
            
        elif user_input.lower() == 'delete':
            try:
                confirm = Prompt.ask("[yellow]Are you sure you want to delete this session?[/yellow]", choices=["yes", "no"], default="no")
                if confirm.lower() == "yes":
                    if db_manager.delete_session(session_id):
                        console.print("[green]Session deleted. Creating a new session...[/green]")
                        session_id = db_manager.create_session()
                        history = []
                        gemini_service.reset_chat()
                    else:
                        console.print("[yellow]Could not delete session from database, but clearing local history.[/yellow]")
                        history = []
                        gemini_service.reset_chat()
            except Exception as e:
                console.print(f"[red]Error deleting session: {e}[/red]")
            continue
            
        elif user_input.lower() == 'clear':
            # Clear history but keep the session
            confirm = Prompt.ask("[yellow]Clear all messages in this session?[/yellow]", choices=["yes", "no"], default="no")
            if confirm.lower() == "yes":
                # Delete messages from database
                if db_manager.is_connected():
                    try:
                        db_manager.client.table("chat_messages").delete().eq("session_id", session_id).execute()
                    except Exception as e:
                        console.print(f"[red]Error clearing messages from database: {e}[/red]")
                
                # Clear local history
                history = []
                gemini_service.reset_chat()
                console.print("[yellow]Conversation history cleared.[/yellow]")
            continue
        
        try:
            # Add to conversation history
            history.append({"role": "user", "content": user_input})
            
            # Save message to database
            db_manager.save_message(session_id, "user", user_input)
            
            # Get response from Gemini with personalization based on history
            response = gemini_service.get_movie_recommendation(user_input, history)
            
            # Add response to history
            history.append({"role": "assistant", "content": response})
            
            # Save response to database
            db_manager.save_message(session_id, "assistant", response)
            
            # Display the response
            console.print("[bold green]Assistant:[/bold green]")
            console.print(Markdown(response))
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


def show_help():
    """Show sample queries the user can ask"""
    help_text = """
    [bold]Sample queries you can ask:[/bold]
    
    - Recommend me some action movies from the 90s
    - What are the best sci-fi movies of all time?
    - I like movies like The Godfather, what should I watch next?
    - Tell me about the movie Inception
    - Who directed Pulp Fiction?
    - What are some good comedy movies for a family night?
    - Recommend me a movie similar to The Matrix
    - What are the top-rated horror movies?
    - I'm in the mood for a romantic comedy, any suggestions?
    
    [bold]Commands:[/bold]
    
    - 'history': Show conversation history
    - 'sessions': List your saved chat sessions
    - 'load <session_id>': Load a specific chat session
    - 'load all': Load memories from all your previous sessions
    - 'rename <new_name>': Rename the current session
    - 'delete': Delete the current session
    - 'clear': Clear the current conversation history
    - 'help': Show this help message
    - 'exit' or 'quit': End the conversation
    """
    console.print(help_text)


def display_history(history):
    """Display the conversation history in a formatted table"""
    if not history:
        console.print("[yellow]No conversation history yet.[/yellow]")
        return
    
    table = Table(title="Conversation History")
    table.add_column("Role", style="cyan")
    table.add_column("Message", style="green")
    
    for i, message in enumerate(history):
        role = message.get("role", "").capitalize()
        content = message.get("content", "")
        
        # Truncate very long messages for display
        if len(content) > 100:
            content = content[:97] + "..."
            
        table.add_row(role, content)
    
    console.print(table)


def display_sessions(sessions):
    """Display available chat sessions"""
    if not sessions:
        console.print("[yellow]No saved sessions found.[/yellow]")
        return
    
    table = Table(title="Your Chat Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Last Updated", style="magenta")
    
    for session in sessions:
        session_id = session.get("id", "")
        # Display truncated ID for readability
        display_id = f"{session_id[:8]}..."
        name = session.get("name", "Unnamed Session")
        
        # Format the timestamp for display
        updated_at = session.get("updated_at", "")
        if updated_at:
            try:
                # Parse ISO format date and reformat
                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                updated_at = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
                
        table.add_row(display_id, name, updated_at)
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--session", help="Load a specific session by ID")
    parser.add_argument("--session-name", help="Name for the chat session")
    
    args = parser.parse_args()
    
    # Start the terminal interface
    run_terminal_interface(
        api_key=args.api_key,
        session_id=args.session,
        session_name=args.session_name
    )


if __name__ == "__main__":
    main()