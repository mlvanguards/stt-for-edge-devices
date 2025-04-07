from typing import Optional
from datetime import datetime
from pydantic import Field, UUID4
from src.config.settings import settings
from src.models.base import BaseModel, TimestampMixin
import uuid

class MessageModel(BaseModel):
    """Core message domain model"""
    id: Optional[UUID4] = Field(default_factory=uuid.uuid4)  # Always generate new UUIDs
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: Optional[float] = None
    conversation_id: str

    class Meta:
        name = "messages"

    def to_mongo(self, **kwargs) -> dict:
        """Override to prevent using conversation_id as _id"""
        parsed = super().to_mongo(**kwargs)
        if "_id" not in parsed:
            parsed["_id"] = str(self.id)
        return parsed


class ConversationModel(BaseModel, TimestampMixin):
    """Core conversation domain model"""
    conversation_id: str
    system_prompt: str = Field(default=settings.conversation.DEFAULT_SYSTEM_PROMPT)
    voice_id: str = Field(default=settings.tts.DEFAULT_VOICE_ID)
    stt_model_id: Optional[str] = Field(default=settings.stt.DEFAULT_STT_MODEL_ID)
    message_count: int = 0
    memory_optimized: bool = False

    class Meta:
        name = "conversations"

