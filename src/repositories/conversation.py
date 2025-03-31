import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.config.settings import settings
from src.repositories.base_mongo import BaseMongoRepository
from src.core.interfaces.repository import IConversationRepository

logger = logging.getLogger(__name__)


class MongoConversationRepository(BaseMongoRepository[Dict[str, Any]], IConversationRepository):
    """
    MongoDB implementation of the ConversationRepository interface.
    """

    def __init__(self):
        """Initialize the repository with collection name."""
        self._collection_name = settings.db.MONGODB_CONVERSATIONS_COLLECTION

    async def _get_collection(self):
        """
        Get the MongoDB collection and ensure indexes are created.
        """
        collection = await super()._get_collection()

        # Create indexes if they don't exist
        await collection.create_index("conversation_id", unique=True)

        return collection

    def _get_id_field_name(self) -> str:
        """
        Get the name of the ID field used by this repository.
        """
        return "conversation_id"

    async def get_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by its conversation_id.

        Args:
            conversation_id: The conversation identifier

        Returns:
            The conversation if found, None otherwise
        """
        try:
            collection = await self._get_collection()
            conversation = await collection.find_one({"conversation_id": conversation_id})

            # Process the conversation document
            if conversation:
                return self._serialize_dates(conversation)

            return None

        except Exception as e:
            logger.error(f"Error getting conversation by ID: {str(e)}")
            return None

    async def increment_message_count(self, conversation_id: str) -> bool:
        """
        Increment the message count for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = await self._get_collection()

            result = await collection.update_one(
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
