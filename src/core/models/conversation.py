from typing import Optional
from datetime import datetime
from pydantic import Field
from src.config.settings import settings
from .base import MongoBaseModel


class MessageModel(MongoBaseModel):
    """Core message domain model"""
    conversation_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance: Optional[float] = None


class ConversationModel(MongoBaseModel):
    """Core conversation domain model"""
    conversation_id: str
    system_prompt: str = Field(default=settings.DEFAULT_SYSTEM_PROMPT)
    voice_id: str = Field(default=settings.DEFAULT_VOICE_ID)
    stt_model_id: Optional[str] = Field(default=settings.DEFAULT_STT_MODEL_ID)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    memory_optimized: bool = False
