import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ConversationRepository(BaseRepository):
    """
    MongoDB implementation of the ConversationRepository interface.
    """

    async def get_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by its conversation_id.
        """
        return await self.get(_id=conversation_id)

    async def increment_message_count(self, conversation_id: str) -> bool:
        """
        Increment the message count for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            True if successful, False otherwise
        """
        try:

            result = await self._collection.update_one(
                {"conversation_id": conversation_id},
                {
                    "$inc": {"message_count": 1},
                    "$set": {"last_updated": datetime.utcnow().isoformat()}
                }
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Error incrementing message count: {str(e)}")
            return False
