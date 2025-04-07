import logging
from typing import Any, Dict, List
import uuid
from src.repositories.base import BaseRepository
from src.models.conversation import MessageModel

logger = logging.getLogger(__name__)

class MessageRepository(BaseRepository):
    # Set the model to the MessageModel from our models
    model = MessageModel

    def __init__(self, db):
        super().__init__(db)

    async def get_by_conversation_id(self, conversation_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            List of messages
        """
        return await self.filter(conversation_id=conversation_id)

    async def update_importance(self, message_id: uuid.UUID, importance: float) -> bool:
        """
        Update the importance score of a message.

        Args:
            message_id: The message identifier
            importance: The new importance score

        Returns:
            True if update successful, False otherwise
        """
        try:
            return await self.update(message_id, {"importance": importance})
        except Exception as e:
            logger.error(f"Error updating message importance: {str(e)}")
            return False

    async def delete_by_conversation_id(self, conversation_id: uuid.UUID) -> int:
        """
        Delete all messages for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            Number of deleted messages
        """
        try:
            result = await self.delete_many(conversation_id=conversation_id)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting messages by conversation: {str(e)}")
            return 0
