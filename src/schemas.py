from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config.settings import settings


class ConversationCreate(BaseModel):
    """Request model for creating a conversation"""

    system_prompt: str = Field(
        default=settings.conversation.DEFAULT_SYSTEM_PROMPT,
        description="System prompt to define the AI assistant's behavior",
    )
    voice_id: Optional[str] = Field(
        default=settings.tts.DEFAULT_VOICE_ID,
        description="Optional voice ID for text-to-speech (defaults to system default)",
    )
    stt_model_id: Optional[str] = Field(
        default=settings.stt.DEFAULT_STT_MODEL_ID,
        description="Optional model ID for speech recognition (defaults to system default)",
    )


class ProcessAudioRequest(BaseModel):
    """Request model for processing audio"""

    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to continue an existing conversation",
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt to override the default"
    )
    voice_id: Optional[str] = Field(
        default=settings.tts.DEFAULT_VOICE_ID,
        description="Optional voice ID for text-to-speech",
    )
    stt_model_id: Optional[str] = Field(
        default=settings.stt.DEFAULT_STT_MODEL_ID,
        description="Optional model ID for speech recognition",
    )

class ConversationResponse(BaseModel):
    """Response model for conversation data"""

    conversation_id: str
    system_prompt: str
    voice_id: str = Field(default=settings.tts.DEFAULT_VOICE_ID)
    stt_model_id: Optional[str] = Field(default=settings.stt.DEFAULT_STT_MODEL_ID)
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
    stt_model_used: Optional[str] = Field(default=settings.stt.DEFAULT_STT_MODEL_ID)
    usage: Dict[str, Any] = {}
    conversation_history: List[Dict[str, Any]] = []
    tts_audio_base64: Optional[str] = None
    memory_stats: Optional[Dict[str, Any]] = None


class TTSResponse(BaseModel):
    """Response model for text-to-speech"""

    audio_base64: str
