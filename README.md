# Movie Recommendation Chatbot

A context-aware movie recommendation system powered by Google's Gemini LLM with persistent conversation history.

## 🍿 Features

- **Personalized Movie Recommendations**: Get movie suggestions based on your preferences, favorite genres, actors, and directors
- **Conversation Memory**: The system remembers your preferences across chat sessions
- **Multiple Interfaces**: Use the terminal application or the web API
- **Persistent Storage**: Conversation history is stored in Supabase for continuous preference learning
- **Context-Aware**: Recommendations improve over time as the system learns your preferences

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Supabase account and API credentials

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Movie-Recommendation.git
   cd Movie-Recommendation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

### Running the Terminal Interface

```bash
python main.py
```

### Running the Web API

```bash
uvicorn app.api.routes:app --reload
```

## 💬 Usage Examples

### Terminal Interface

The terminal interface supports various commands and queries:

- Ask for recommendations: "Recommend me movies like Inception"
- Get information about a movie: "Tell me about The Godfather"
- Find movies by actor: "What are some good Tom Hanks movies?"
- Find movies by director: "I like Christopher Nolan films"
- Get genre recommendations: "I'm in the mood for a sci-fi movie"
- Exit the chat: "exit" or "quit"

### API Endpoints

- `GET /api/health`: Health check endpoint
- `POST /api/chat`: Chat with the recommendation system
  - Request body:
    ```json
    {
      "query": "Recommend me sci-fi movies with time travel",
      "conversation_history": [] // Optional previous conversation
    }
    ```

## 🏗️ Project Structure

```
├── main.py                  # Terminal interface entry point
├── requirements.txt         # Project dependencies
├── app/                    
│   └── api/                
│       ├── __init__.py      
│       ├── routes.py        # API setup for Flask
│       └── fast_api_routes.py # FastAPI routes
├── models/               
│   └── schemas.py           # Data models
├── services/            
│   └── gemini_service.py    # Gemini LLM integration
└── utils/               
    └── db_utils.py          # Database utilities for Supabase
```

## 🧠 How It Works

1. User submits a query about movies
2. The system retrieves conversation history from Supabase
3. The Gemini LLM processes the query along with conversation context
4. The LLM generates personalized recommendations based on the user's preferences
5. Responses and conversation history are stored in Supabase for future context

## 🔧 Advanced Configuration

You can customize the system prompt and model parameters in the `services/gemini_service.py` file.

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- [Google Generative AI](https://ai.google/discover/generative-ai/) for Gemini LLM
- [Supabase](https://supabase.com/) for database storage
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework