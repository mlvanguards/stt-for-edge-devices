from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Tuple

T = TypeVar('T')


class IRepository(Generic[T], ABC):
    """Base repository interface for data access."""

    @abstractmethod
    async def create(self, entity: Dict[str, Any]) -> Optional[T]:
        """
        Create a new entity.

        Args:
            entity: Entity data

        Returns:
            Created entity or None
        """
        pass

    @abstractmethod
    async def read(self, id: str) -> Optional[T]:
        """
        Read entity by ID.

        Args:
            id: Entity ID

        Returns:
            Entity if found or None
        """
        pass

    @abstractmethod
    async def update(self, id: str, entity: Dict[str, Any]) -> bool:
        """
        Update an entity.

        Args:
            id: Entity ID
            entity: Updated data

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete an entity.

        Args:
            id: Entity ID

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def list(self, skip: int = 0, limit: int = 100, filters: Dict[str, Any] = None) -> List[T]:
        """
        List entities with pagination and filtering.

        Args:
            skip: Number to skip
            limit: Max to return
            filters: Optional filters

        Returns:
            List of entities
        """
        pass


class IConversationRepository(IRepository[Dict[str, Any]], ABC):
    """Repository interface for conversations."""

    @abstractmethod
    async def get_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation by conversation_id.

        Args:
            conversation_id: The conversation identifier

        Returns:
            Conversation if found or None
        """
        pass

    @abstractmethod
    async def increment_message_count(self, conversation_id: str) -> bool:
        """
        Increment message count for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            Success status
        """
        pass


class IMessageRepository(IRepository[Dict[str, Any]], ABC):
    """Repository interface for messages."""

    @abstractmethod
    async def get_by_conversation_id(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            List of messages
        """
        pass

    @abstractmethod
    async def update_importance(self, message_id: str, importance: float) -> bool:
        """
        Update message importance score.

        Args:
            message_id: The message identifier
            importance: New importance score

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def delete_by_conversation_id(self, conversation_id: str) -> int:
        """
        Delete all messages for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            Number of deleted messages
        """
        pass


class IMemoryRepository(IRepository[Dict[str, Any]], ABC):
    """Repository interface for memory summaries."""

    @abstractmethod
    async def get_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get memory summary for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            Memory summary if found or None
        """
        pass

    @abstractmethod
    async def update_summary(self, conversation_id: str, summary_text: str) -> bool:
        """
        Update or create summary for a conversation.

        Args:
            conversation_id: The conversation identifier
            summary_text: New summary

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def delete_by_conversation_id(self, conversation_id: str) -> bool:
        """
        Delete memory summary for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            Success status
        """
        pass


class IAudioRepository(IRepository[Dict[str, Any]], ABC):
    """Repository interface for audio files."""

    @abstractmethod
    async def save_audio(
            self,
            audio_content: bytes,
            content_type: str,
            conversation_id: Optional[str] = None,
            ttl_hours: int = 24
    ) -> Optional[str]:
        """
        Save audio content.

        Args:
            audio_content: Raw audio bytes
            content_type: MIME type
            conversation_id: Optional conversation ID
            ttl_hours: Time-to-live in hours

        Returns:
            Audio ID if saved or None
        """
        pass

    @abstractmethod
    async def get_audio(self, audio_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Get audio content by ID.

        Args:
            audio_id: Audio identifier

        Returns:
            Tuple of (audio_content, content_type) or (None, None)
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Delete expired audio files.

        Returns:
            Number of deleted files
        """
        pass
