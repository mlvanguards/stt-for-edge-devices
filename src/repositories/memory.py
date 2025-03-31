import logging
from datetime import datetime
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import PyMongoError

from src.config.settings import settings
from src.core.interfaces.repository import IMemoryRepository
from src.repositories.base_mongo import BaseMongoRepository

logger = logging.getLogger(__name__)


class MongoMemoryRepository(BaseMongoRepository[Dict[str, Any]], IMemoryRepository):
    """
    MongoDB implementation of the MemoryRepository interface for conversation memory summaries.
    """

    def __init__(self):
        """Initialize the repository with collection name."""
        self._collection_name = settings.db.MONGODB_MEMORY_COLLECTION

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """
        Get the MongoDB collection and ensure indexes are created.
        """
        collection = await super()._get_collection()

        # Create indexes if they don't exist - using custom name to avoid conflicts
        try:
            await collection.create_index(
                [("conversation_id", 1)],
                unique=True,
                name="memory_conversation_id_unique"
            )
            logger.debug("Memory repository index created or verified")
        except PyMongoError as e:
            # If index already exists with different options, log but continue
            logger.warning(f"Error creating memory repository index: {str(e)}")

        return collection

    async def get_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get memory summary for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            The memory summary if found, None otherwise
        """
        try:
            collection = await self._get_collection()

            memory = await collection.find_one({"conversation_id": conversation_id})

            # Process the memory document
            if memory:
                memory = self._serialize_dates(memory)
                memory["_id"] = str(memory["_id"])

            return memory

        except PyMongoError as e:
            logger.error(f"MongoDB error getting memory summary by conversation: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting memory summary by conversation: {str(e)}")
            return None

    async def update_summary(self, conversation_id: str, summary_text: str) -> bool:
        """
        Update or create the summary for a conversation.

        Args:
            conversation_id: The conversation identifier
            summary_text: The new summary text

        Returns:
            True if update successful, False otherwise
        """
        try:
            collection = await self._get_collection()

            timestamp = datetime.utcnow().isoformat()

            # Check if summary exists
            existing = await collection.find_one({"conversation_id": conversation_id})

            if existing:
                # Update existing summary
                result = await collection.update_one(
                    {"conversation_id": conversation_id},
                    {
                        "$set": {
                            "summary": summary_text,
                            "updated_at": timestamp
                        }
                    }
                )
                return result.modified_count > 0
            else:
                # Create new summary
                result = await collection.insert_one({
                    "conversation_id": conversation_id,
                    "summary": summary_text,
                    "created_at": timestamp,
                    "updated_at": timestamp
                })
                return result.acknowledged

        except PyMongoError as e:
            logger.error(f"MongoDB error updating conversation summary: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating conversation summary: {str(e)}")
            return False

    async def delete_by_conversation_id(self, conversation_id: str) -> bool:
        """
        Delete memory summary for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            collection = await self._get_collection()

            result = await collection.delete_one({"conversation_id": conversation_id})

            return result.deleted_count > 0

        except PyMongoError as e:
            logger.error(f"MongoDB error deleting memory summary by conversation: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting memory summary by conversation: {str(e)}")
            return False
