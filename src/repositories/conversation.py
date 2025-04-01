import logging
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

from src.repositories.base import BaseRepository
from src.models.conversation import ConversationModel

logger = logging.getLogger(__name__)

class ConversationRepository(BaseRepository):
    # Set the model to the ConversationModel from our models
    model = ConversationModel

    def __init__(self, db):
        super().__init__(db)

    async def get_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by its conversation_id.
        """
        try:
            collection = self._collection  # Use the existing collection directly
            conversation = await collection.find_one({"conversation_id": conversation_id})

            if conversation:
                # Convert ObjectId to string if needed
                if "_id" in conversation and not isinstance(conversation["_id"], str):
                    conversation["_id"] = str(conversation["_id"])

                # Ensure there's a valid UUID for Pydantic
                if "id" not in conversation:
                    conversation["id"] = uuid.uuid4()

            return conversation
        except Exception as e:
            logger.error(f"Error fetching conversation: {str(e)}")
            return None

    async def increment_message_count(self, conversation_id: str) -> bool:
        """
        Increment the message count for a conversation.
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
