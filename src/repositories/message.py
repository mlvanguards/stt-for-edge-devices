import logging
from typing import Any, Dict, List

from src.config.settings import settings
from src.repositories.base_mongo import BaseMongoRepository
from src.core.interfaces.repository import IMessageRepository

logger = logging.getLogger(__name__)


class MongoMessageRepository(BaseMongoRepository[Dict[str, Any]], IMessageRepository):
    """
    MongoDB implementation of the MessageRepository interface.
    """

    def __init__(self):
        """Initialize the repository with collection name."""
        self._collection_name = settings.db.MONGODB_MESSAGES_COLLECTION

    async def _get_collection(self):
        """
        Get the MongoDB collection and ensure indexes are created.
        """
        collection = await super()._get_collection()

        # Create indexes if they don't exist
        await collection.create_index("conversation_id")
        await collection.create_index([("conversation_id", 1), ("timestamp", 1)])
        await collection.create_index([("conversation_id", 1), ("importance", -1)])

        return collection

    async def get_by_conversation_id(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            List of messages
        """
        try:
            collection = await self._get_collection()

            cursor = collection.find({"conversation_id": conversation_id}).sort("timestamp", 1)
            messages = await cursor.to_list(length=None)

            # Process messages
            result = []
            for message in messages:
                message = self._serialize_dates(message)
                message["_id"] = str(message["_id"])
                result.append(message)

            return result

        except Exception as e:
            logger.error(f"Error getting messages by conversation: {str(e)}")
            return []

    async def update_importance(self, message_id: str, importance: float) -> bool:
        """
        Update the importance score of a message.

        Args:
            message_id: The message identifier (MongoDB ObjectId as string)
            importance: The new importance score

        Returns:
            True if update successful, False otherwise
        """
        try:
            return await self.update(message_id, {"importance": importance})
        except Exception as e:
            logger.error(f"Error updating message importance: {str(e)}")
            return False

    async def delete_by_conversation_id(self, conversation_id: str) -> int:
        """
        Delete all messages for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            Number of deleted messages
        """
        try:
            collection = await self._get_collection()

            result = await collection.delete_many({"conversation_id": conversation_id})

            return result.deleted_count

        except Exception as e:
            logger.error(f"Error deleting messages by conversation: {str(e)}")
            return 0
