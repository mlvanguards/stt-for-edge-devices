from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.config.settings import settings


class ConversationResponse(BaseModel):
    """Response model for conversation data"""
    conversation_id: str
    system_prompt: str
    voice_id: str = Field(default=settings.DEFAULT_VOICE_ID)
    stt_model_id: Optional[str] = Field(default=settings.DEFAULT_STT_MODEL_ID)
    messages: List[Dict[str, Any]] = []


class ConversationListResponse(BaseModel):
    """Response model for listing conversations"""
    total: int
    conversations: List[Dict[str, Any]]
    page: int
    limit: int
    pages: int


class ChatResponse(BaseModel):
    """Response model for chat processing"""
    conversation_id: str
    transcription: str
    raw_transcription: str
    segment_transcriptions: List[Dict[str, Any]]
    num_segments: int
    response: str
    model: str
    stt_model_used: Optional[str] = Field(default=settings.DEFAULT_STT_MODEL_ID)
    usage: Dict[str, Any] = {}
    conversation_history: List[Dict[str, Any]] = []
    tts_audio_base64: Optional[str] = None
    memory_stats: Optional[Dict[str, Any]] = None


class TTSResponse(BaseModel):
    """Response model for text-to-speech"""
    audio_base64: str
