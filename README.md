# Speech-to-Text for Edge Devices 🎤

A lightweight, optimized speech recognition application designed for child voice recognition with conversation continuity, text-to-speech feedback, and MongoDB persistence.

## Demo

[![Application Demo](https://img.youtube.com/vi/9-eIeuKasx0/0.jpg)](https://www.youtube.com/watch?v=9-eIeuKasx0)

## 🌟 Features

- **Conversational Context**: Maintains conversation history across requests
- **Text-to-Speech**: Converts AI responses to spoken audio using ElevenLabs
- **Edge Device Profiling**: Tools to measure and optimize for low-resource environments
- **MongoDB Integration**: Persistent storage of conversations and transcriptions
- **Serverless Ready**: Deployable with [Genezio](https://genez.io/)

## 🔑 API Key Management

This application requires users to provide their own API keys for the following services:

- **Hugging Face**: For speech recognition (STT) models
- **OpenAI**: For chat functionality (GPT models)
- **ElevenLabs**: For text-to-speech synthesis

### Submitting API Keys

The application provides endpoints to submit your API keys which will be stored in memory for the duration of the session:

#### Submit all API keys at once:

```http
POST /api-keys/submit
```

Request body:
```json
{
  "huggingface_token": "your_huggingface_token",
  "openai_api_key": "your_openai_api_key",
  "elevenlabs_api_key": "your_elevenlabs_api_key"
}
```

#### Check API key status:

```http
GET /api-keys/status
```

#### Reset API keys:

```http
DELETE /api-keys/reset
```

> **Important**: The application will not function correctly without these API keys. You must provide your own valid API keys before using the core features.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Poetry (recommended) or pip
- MongoDB database

### Installation

#### Using Poetry (recommended)

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

#### Using Pip

```bash
# Install requirements
pip install -r requirements.txt
```

### Running Locally

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload
```

Visit `http://localhost:8000/docs` to access the Swagger UI and test the API.

## 📋 API Endpoints

### API Key Management

- **POST /api-keys/submit** - Submit all API keys at once
- **POST /api-keys/huggingface** - Submit HuggingFace token
- **POST /api-keys/openai** - Submit OpenAI API key
- **POST /api-keys/elevenlabs** - Submit ElevenLabs API key
- **GET /api-keys/status** - Check API key status
- **DELETE /api-keys/reset** - Reset all API keys

### Speech-to-Text + Chat

- **POST /chat** - Process audio, maintain conversation context, get AI response
  - Accepts audio files with transcription requests
  - Maintains conversation history
  - Returns text responses and optional TTS audio

### Conversation Management

- **POST /create_conversation** - Create a new conversation with system prompt
- **GET /conversations/{conversation_id}** - Get conversation history
- **DELETE /conversations/{conversation_id}** - Delete a conversation
- **GET /conversations** - List all conversations with pagination

### Text-to-Speech

- **POST /tts_only** - Convert text to speech using ElevenLabs
- **GET /available_voices** - Get available voice options

## 🏗️ Project Structure

```
├── genezio.yaml          # Genezio serverless configuration
├── requirements.txt      # Python dependencies for deployment
├── pyproject.toml        # Poetry configuration
├── .env.example          # Environment variables template 
├── data/                 # Example data and results
└── src/
    ├── api/              # FastAPI server and routes
    │   ├── routes/       # API endpoint implementations
    │   └── main.py       # FastAPI application entry point
    ├── asr/              # Automatic speech recognition
    ├── config/           # Configuration settings
    ├── core/             # Core functionality
    │   ├── database.py   # MongoDB integration
    │   ├── chat.py       # OpenAI/ChatGPT integration
    │   └── speech/       # Speech processing utilities
    ├── data/             # Data processing utilities
    ├── models/           # Pydantic data models
    ├── resource_testing/ # Edge device profiling tools
    └── utils/            # Utility functions
        ├── api_keys_service.py # API keys utility functions
        └── audio_handling.py   # Audio processing utilities
```

## 🔍 Edge Device Profiling

This project includes specialized tools to measure performance on edge devices:

```bash
# Run the edge device profiler
python src/resource_testing/run_edge_profiler.py
```

The profiler measures:
- Real-time processing factor
- Memory usage
- CPU utilization
- Battery consumption (when available)
- Edge device suitability score

## 🚢 Deployment with Genezio

1. Install Genezio CLI:
   ```bash
   npm install -g genezio
   ```

2. Log in to Genezio:
   ```bash
   genezio login
   ```

3. Deploy the application:
   ```bash
   genezio deploy
   ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co/) for speech recognition models
- [OpenAI](https://openai.com/) for conversational AI
- [ElevenLabs](https://elevenlabs.io/) for text-to-speech
- [Genezio](https://genez.io/) for serverless deployment