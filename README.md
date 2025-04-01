<p align="center">   
   <h1 align="center">Speech-to-Text for Edge Devices 🎤</h1>
   <h3 align="center">A lightweight, optimized speech recognition application designed for child voice recognition with conversation continuity, text-to-speech feedback, and MongoDB persistence.</h3>
</p>

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


> **Important**: The application will not function correctly without these API keys. You must create a .env with your own valid API keys before using the core features.

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
uvicorn src.main:app --reload
```

Visit `http://localhost:8000/docs` to access the Swagger UI and test the API.

## 🖥️ Frontend Interface

The frontend app is built as a separate project to keep the backend lightweight and focused.

👉 You can find the full frontend repository here: 🔗 [stt-for-edge-devices-frontend](https://github.com/mlvanguards/stt-for-edge-devices-frontend)

This React-based interface connects directly to the FastAPI backend, allows converations with different model/voices configuration, transcriptions and receives AI responses in both audio and text.

To run the frontend locally:

```bash
# Clone the repo
git clone https://github.com/mlvanguards/stt-for-edge-devices-frontend.git
cd stt-for-edge-devices-frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

Once running, the frontend will connect to your local FastAPI server at http://localhost:8000 by default.

## 📋 API Endpoints

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
stt-for-edge-devices/
├── README.md
├── genezio.yaml               # Serverless deployment configuration
├── pyproject.toml             # Poetry project configuration
├── requirements.txt           # Pip dependencies
├── .env.example               # Environment variables template
├── data/                      # Example dataset (transcriptions-test.json)
├── scripts/                   # Auxiliary scripts and notebooks
│   ├── automatic_transcription.py  # Automatic ASR processing script
│   └── fine_tuning_and_optimization.ipynb  # Fine-tuning & optimization notebook
└── src/                       # Main application code
    ├── main.py              # FastAPI application entry point
    ├── db.py                # MongoDB connection management
    ├── dependencies.py      # Dependency injection setup
    ├── errors.py            # Custom exception definitions
    ├── schemas.py           # Pydantic request/response models
    ├── config/              # Application settings
    │   └── settings.py
    ├── api/                 # API routes
    │   └── v1/
    │       ├── chat.py
    │       ├── conversations.py
    │       └── tts.py
    ├── asr/                 # Speech recognition modules
    │   ├── asr_processor.py
    │   ├── base.py
    │   └── speech_brain.py
    ├── gateways/            # External API integrations (HuggingFace, OpenAI, ElevenLabs)
    ├── models/              # Data models
    ├── repositories/        # Database repositories (conversations, messages, audio, memory)
    ├── services/            # Business logic (chat, conversation, ASR, TTS, memory)
    ├── resource_testing/    # Edge device profiling tools
    │   └── run_edge_profiler.py
    └── utils/               # Utility modules (audio processing, data normalization, etc.)

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
