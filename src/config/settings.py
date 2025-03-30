from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic_settings import BaseSettings


class BaseAppSettings(BaseSettings):
    """Base settings class with common configuration."""

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class APISettings(BaseAppSettings):
    """API-related settings."""

    API_VERSION: str = "1.0.0"
    API_TITLE: str = "STT and Chat API with Conversation Memory, TTS, and MongoDB"
    API_DESCRIPTION: str = """
    Speech-to-text transcription, conversation management with GPT models and short-term memory, 
    and text-to-speech synthesis API optimized for edge devices.
    """

    # CORS settings
    CORS_ALLOW_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]


class AuthSettings(BaseAppSettings):
    """Authentication and API keys settings."""

    HUGGINGFACE_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None


class STTSettings(BaseAppSettings):
    """Speech-to-text related settings."""

    HUGGINGFACE_API_URL: str = "https://api-inference.huggingface.co/models"
    AVAILABLE_STT_MODELS: List[Dict[str, str]] = [
        {
            "id": "StefanStefan/Wav2Vec-100-CSR",
            "name": "Wav2Vec-100-CSR (Default)",
            "description": "Standard model with good accuracy",
        },
        {
            "id": "StefanStefan/Wav2Vec-100-CSR-Quantized",
            "name": "Wav2Vec-100-CSR-Quantized",
            "description": "Quantized model for faster inference with slight accuracy trade-off",
        },
        {
            "id": "StefanStefan/Wav2Vec-100-CSR-KD",
            "name": "Wav2Vec-100-CSR-KD",
            "description": "Knowledge distilled model with improved efficiency",
        },
        {
            "id": "StefanStefan/Wav2Vec-100-CSR-Distilled-Quantized",
            "name": "Wav2Vec-100-CSR-Distilled-Quantized",
            "description": "Distilled and quantized model for maximum efficiency",
        },
    ]
    DEFAULT_STT_MODEL_ID: str = "StefanStefan/Wav2Vec-100-CSR"
    SPEECH_RECOGNITION_RETRIES: int = 3
    SPEECH_RECOGNITION_BACKOFF_FACTOR: int = 2


class OpenAISettings(BaseAppSettings):
    """OpenAI API settings."""

    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    GPT_MODEL: str = "gpt-4o"
    GPT_TEMPERATURE: float = 0.7
    GPT_MAX_TOKENS: int = 500


class MemorySettings(BaseAppSettings):
    """Conversation memory settings."""

    MEMORY_ENABLED: bool = True
    MEMORY_MAX_MESSAGES: int = 15
    MEMORY_SUMMARIZE_THRESHOLD: int = 5


class TTSSettings(BaseAppSettings):
    """Text-to-speech related settings."""

    ELEVENLABS_API_URL: str = "https://api.elevenlabs.io/v1/text-to-speech"
    DEFAULT_VOICE_ID: str = "cgSgspJ2msm6clMCkdW9"
    TTS_MODEL_ID: str = "eleven_turbo_v2"
    TTS_DEFAULT_SETTINGS: Dict[str, Any] = {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    }


class DatabaseSettings(BaseAppSettings):
    """Database connection settings."""

    MONGODB_URI: Optional[str] = None
    MONGODB_DB: str = "stt-app-db"
    MONGODB_CONVERSATIONS_COLLECTION: str = "conversations"
    MONGODB_MESSAGES_COLLECTION: str = "messages"
    MONGODB_MEMORY_COLLECTION: str = "memory_summaries"
    MONGODB_SERVER_SELECTION_TIMEOUT_MS: int = 5000
    MONGODB_CONNECT_TIMEOUT_MS: int = 10000
    MONGODB_MAX_POOL_SIZE: int = 10
    MONGODB_MIN_POOL_SIZE: int = 1
    MONGODB_MAX_IDLE_TIME_MS: int = 30000
    MONGODB_RETRY_WRITES: bool = True


class AudioSettings(BaseAppSettings):
    """Audio processing settings."""

    AUDIO_SEGMENT_DURATION: int = 10
    ALLOWED_AUDIO_CONTENT_TYPES: List[str] = [
        "audio/wav",
        "audio/mpeg",
        "audio/x-wav",
        "audio/webm",
    ]
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_FORMAT: str = "wav"


class ConversationSettings(BaseAppSettings):
    """Conversation management settings."""

    DEFAULT_SYSTEM_PROMPT: str = """
    You are a teacher having casual conversation with kids below the age of 12. 
    Do not try to correct their typos just keep the conversation going with them.
    Use your memory of previous conversations to make the interaction more natural.
    """
    MAX_CONVERSATION_HISTORY: int = 100


class DataSettings(BaseAppSettings):
    """Data processing and training settings."""

    DATA_TRAIN_SIZE: float = 0.7
    DATA_VAL_SIZE: float = 0.15
    DATA_TEST_SIZE: float = 0.15
    DATA_RANDOM_SEED: int = 42
    DATA_MAX_AUDIO_DURATION: float = 10.0
    DATA_AUDIO_EXTENSIONS: List[str] = [".wav", ".flac", ".mp3"]
    DATA_MIN_WORD_LENGTH: int = 2
    DATA_MAX_WORD_FREQUENCY: int = 3
    DATA_SHORT_WORD_THRESHOLD: float = 0.2
    DATA_KAGGLE_DATASET: str = "mirfan899/kids-speech-dataset"
    DATA_OUTPUT_DIR: str = "dataset_kaggle"


class ModelSettings(BaseAppSettings):
    """Model training and inference settings."""

    MODEL_DEVICE: str = "cpu"
    MODEL_NUM_CORES: int = 6
    MODEL_BATCH_SIZE: int = 10
    MODEL_SAVE_INTERVAL: int = 50


class TestingSettings(BaseAppSettings):
    """Testing and benchmarking settings."""

    TESTING_SAMPLING_INTERVAL: float = 0.1
    TESTING_DEFAULT_NUM_REPEATS: int = 3
    TESTING_DEFAULT_DEVICE: str = "cpu"
    TESTING_EDGE_MEMORY_THRESHOLD_MB: float = 2000.0
    TESTING_EDGE_CPU_THRESHOLD_PERCENT: float = 50.0
    TESTING_BATTERY_DRAIN_THRESHOLD_PERCENT: float = 1.0


class Settings(BaseAppSettings):
    """Main settings class that combines all specialized settings."""

    api: APISettings = APISettings()
    auth: AuthSettings = AuthSettings()
    stt: STTSettings = STTSettings()
    openai: OpenAISettings = OpenAISettings()
    memory: MemorySettings = MemorySettings()
    tts: TTSSettings = TTSSettings()
    db: DatabaseSettings = DatabaseSettings()
    audio: AudioSettings = AudioSettings()
    conversation: ConversationSettings = ConversationSettings()
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    testing: TestingSettings = TestingSettings()


# Create global settings instance
settings = Settings()
