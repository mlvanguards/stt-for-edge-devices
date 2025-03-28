from typing import Optional
from pydantic import BaseModel, Field
from src.config.settings import settings


class ConversationCreate(BaseModel):
    """Request model for creating a conversation"""
    system_prompt: str = Field(
        default=settings.DEFAULT_SYSTEM_PROMPT,
        description="System prompt to define the AI assistant's behavior"
    )
    voice_id: Optional[str] = Field(
        default=settings.DEFAULT_VOICE_ID,
        description="Optional voice ID for text-to-speech (defaults to system default)"
    )
    stt_model_id: Optional[str] = Field(
        default=settings.DEFAULT_STT_MODEL_ID,
        description="Optional model ID for speech recognition (defaults to system default)"
    )


class ProcessAudioRequest(BaseModel):
    """Request model for processing audio"""
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to continue an existing conversation"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to override the default"
    )
    voice_id: Optional[str] = Field(
        default=settings.DEFAULT_VOICE_ID,
        description="Optional voice ID for text-to-speech"
    )
    stt_model_id: Optional[str] = Field(
        default=settings.DEFAULT_STT_MODEL_ID,
        description="Optional model ID for speech recognition"
    )
