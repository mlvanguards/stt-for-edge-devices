import logging
from datetime import datetime
from typing import Any, Dict, Optional

from pymongo.errors import PyMongoError

from src.config.settings import settings
from src.repositories.base import BaseRepository
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class MemoryModel(BaseModel):
    class Meta:
        name = settings.db.MONGODB_MEMORY_COLLECTION

class MemoryRepository(BaseRepository):
    model = MemoryModel

    def __init__(self, db):
        super().__init__(db)

    async def _get_collection(self):
        collection = self._collection
        try:
            await collection.create_index(
                [("conversation_id", 1)],
                unique=True,
                name="memory_conversation_id_unique"
            )
            logger.debug("Memory repository index created or verified")
        except PyMongoError as e:
            logger.warning(f"Error creating memory repository index: {str(e)}")
        return collection

    async def get_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        try:
            collection = await self._get_collection()
            memory = await collection.find_one({"conversation_id": conversation_id})
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
        try:
            collection = await self._get_collection()
            timestamp = datetime.utcnow().isoformat()
            existing = await collection.find_one({"conversation_id": conversation_id})
            if existing:
                result = await collection.update_one(
                    {"conversation_id": conversation_id},
                    {"$set": {"summary": summary_text, "updated_at": timestamp}}
                )
                return result.modified_count > 0
            else:
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
