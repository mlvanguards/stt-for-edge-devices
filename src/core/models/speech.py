from pydantic import BaseModel, Field
from src.config.settings import settings


class TTSSettings(BaseModel):
    """Text-to-speech settings"""
    voice_id: str
    model_id: str = Field(default=settings.TTS_MODEL_ID)
    stability: float = Field(default=settings.TTS_DEFAULT_SETTINGS["stability"])
    similarity_boost: float = Field(default=settings.TTS_DEFAULT_SETTINGS["similarity_boost"])
    style: float = Field(default=settings.TTS_DEFAULT_SETTINGS["style"])
    use_speaker_boost: bool = Field(default=settings.TTS_DEFAULT_SETTINGS["use_speaker_boost"])
