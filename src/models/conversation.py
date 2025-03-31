from typing import Optional
from datetime import datetime
from pydantic import Field, UUID4
from src.config.settings import settings
from src.models.base import BaseModel, TimestampMixin


class MessageModel(BaseModel):
    """Core message domain model"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: Optional[float] = None
    conversation_id: UUID4

    class Meta:
        name = "messages"


class ConversationModel(BaseModel, TimestampMixin):
    """Core conversation domain model"""
    system_prompt: str = Field(default=settings.DEFAULT_SYSTEM_PROMPT)
    voice_id: str = Field(default=settings.DEFAULT_VOICE_ID)
    stt_model_id: Optional[str] = Field(default=settings.DEFAULT_STT_MODEL_ID)
    message_count: int = 0
    memory_optimized: bool = False

    class Meta:
        name = "conversations"
