from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from src.config.settings import DEFAULT_VOICE_ID, DEFAULT_SYSTEM_PROMPT, DEFAULT_STT_MODEL_ID
from src.models.common import MongoBaseModel

class MessageModel(MongoBaseModel):
    """Message document model"""
    conversation_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationModel(MongoBaseModel):
    """Conversation document model"""
    conversation_id: str
    system_prompt: str
    voice_id: str = DEFAULT_VOICE_ID
    stt_model_id: str = DEFAULT_STT_MODEL_ID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# API Request Models
class ConversationCreate(BaseModel):
    """Request model for creating a conversation"""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    voice_id: Optional[str] = DEFAULT_VOICE_ID
    stt_model_id: Optional[str] = DEFAULT_STT_MODEL_ID


class ProcessAudioRequest(BaseModel):
    """Request model for processing audio"""
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None
    voice_id: Optional[str] = DEFAULT_VOICE_ID
    stt_model_id: Optional[str] = DEFAULT_STT_MODEL_ID


# API Response Models
class ConversationResponse(BaseModel):
    """Response model for conversation data"""
    conversation_id: str
    system_prompt: str
    voice_id: str
    stt_model_id: Optional[str] = DEFAULT_STT_MODEL_ID
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
    stt_model_used: Optional[str] = DEFAULT_STT_MODEL_ID
    usage: Dict[str, Any] = {}
    conversation_history: List[Dict[str, Any]] = []
    tts_audio_base64: Optional[str] = None


class TTSResponse(BaseModel):
    """Response model for text-to-speech"""
    audio_base64: str
