import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API information
API_VERSION = "1.0.0"
API_TITLE = "STT and Chat API with Conversation Memory, TTS, and MongoDB"
API_DESCRIPTION = """
Speech-to-text transcription, conversation management with GPT models and short-term memory, 
and text-to-speech synthesis API optimized for edge devices.
"""

# API Configurations
# Hugging Face API configuration
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# Available STT Models
AVAILABLE_STT_MODELS = [
    {
        "id": "StefanStefan/Wav2Vec-100-CSR",
        "name": "Wav2Vec-100-CSR (Default)",
        "description": "Standard model with good accuracy"
    },
    {
        "id": "StefanStefan/Wav2Vec-100-CSR-Quantized",
        "name": "Wav2Vec-100-CSR-Quantized",
        "description": "Quantized model for faster inference with slight accuracy trade-off"
    },
    {
        "id": "StefanStefan/Wav2Vec-100-CSR-KD",
        "name": "Wav2Vec-100-CSR-KD",
        "description": "Knowledge distilled model with improved efficiency"
    },
    {
        "id": "StefanStefan/Wav2Vec-100-CSR-Distilled-Quantized",
        "name": "Wav2Vec-100-CSR-Distilled-Quantized",
        "description": "Distilled and quantized model for maximum efficiency"
    }
]

# Default STT model ID (used when no model is specified)
DEFAULT_STT_MODEL_ID = "StefanStefan/Wav2Vec-100-CSR"
HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models"

# OpenAI/GPT configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GPT_MODEL = "gpt-4o"
GPT_TEMPERATURE = 0.7
GPT_MAX_TOKENS = 500

# Memory settings
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "15"))
MEMORY_SUMMARIZE_THRESHOLD = int(os.getenv("MEMORY_SUMMARIZE_THRESHOLD", "5"))


# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
DEFAULT_VOICE_ID = "cgSgspJ2msm6clMCkdW9"

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "stt-app-db")
MONGODB_CONVERSATIONS_COLLECTION = "conversations"
MONGODB_MESSAGES_COLLECTION = "messages"
MONGODB_MEMORY_COLLECTION = "memory_summaries"

# Audio processing configuration
AUDIO_SEGMENT_DURATION = 10  # seconds
ALLOWED_AUDIO_CONTENT_TYPES = ["audio/wav", "audio/mpeg", "audio/x-wav", "audio/webm"]

# Speech recognition settings
SPEECH_RECOGNITION_RETRIES = 3
SPEECH_RECOGNITION_BACKOFF_FACTOR = 2

# TTS settings
TTS_MODEL_ID = "eleven_turbo_v2"
TTS_DEFAULT_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True
}

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are a teacher having casual conversation with kids below the age of 12. 
Do not try to correct their typos just keep the conversation going with them.
Use your memory of previous conversations to make the interaction more natural.
"""

# CORS settings
CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]