import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from bson.objectid import ObjectId
from bson.binary import Binary
from motor.motor_asyncio import AsyncIOMotorCollection

from src.repositories.base_mongo import BaseMongoRepository
from src.core.interfaces.repository import IAudioRepository
from src.core.utils.audio.audio_handling import AudioProcessor

logger = logging.getLogger(__name__)


class MongoAudioRepository(BaseMongoRepository[Dict[str, Any]], IAudioRepository):
    """
    MongoDB implementation of the AudioRepository interface.
    Uses the centralized AudioProcessor for audio-related operations.
    """

    def __init__(self, audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize the repository with collection name and audio processor.

        Args:
            audio_processor: Optional AudioProcessor instance
        """
        self._collection_name = "audio_files"  # Not in settings, add if needed
        self.audio_processor = audio_processor or AudioProcessor()

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """
        Get the MongoDB collection and ensure indexes are created.
        """
        collection = await super()._get_collection()

        # Create indexes if they don't exist
        await collection.create_index("conversation_id")
        await collection.create_index("expires_at", expireAfterSeconds=0)  # TTL index

        return collection

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
            content_type: MIME type of the audio
            conversation_id: Optional conversation ID to associate with
            ttl_hours: Time-to-live in hours for automatic cleanup

        Returns:
            The audio ID if saved successfully, None otherwise
        """
        try:
            # Validate content type using audio processor
            if not self.audio_processor.validate_content_type(content_type):
                logger.warning(f"Invalid content type: {content_type}")
                return None

            # Get audio duration if possible
            duration = None
            try:
                duration = self.audio_processor.get_audio_duration(audio_content)
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {str(e)}")

            # Create audio document
            audio_doc = {
                "content": Binary(audio_content),  # Store as Binary BSON type
                "content_type": content_type,
                "size_bytes": len(audio_content),
                "duration": duration,
                "conversation_id": conversation_id,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(hours=ttl_hours)
            }

            # Create using base method
            result = await self.create(audio_doc)

            if result:
                audio_id = str(result["_id"])
                logger.info(f"Saved audio file {audio_id} ({len(audio_content)} bytes)")
                return audio_id

            return None

        except Exception as e:
            logger.error(f"Error saving audio to MongoDB: {str(e)}")
            return None

    async def get_audio(self, audio_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Get audio content by ID.

        Args:
            audio_id: The audio identifier

        Returns:
            Tuple of (audio_content, content_type) if found, (None, None) otherwise
        """
        try:
            collection = await self._get_collection()

            # Find audio document
            try:
                object_id = ObjectId(audio_id)
            except Exception:
                logger.warning(f"Invalid ObjectId: {audio_id}")
                return None, None

            audio_doc = await collection.find_one({"_id": object_id})

            if not audio_doc:
                logger.warning(f"Audio file {audio_id} not found")
                return None, None

            # Return audio content and type
            return audio_doc["content"], audio_doc["content_type"]

        except Exception as e:
            logger.error(f"Error retrieving audio from MongoDB: {str(e)}")
            return None, None

    async def cleanup_expired(self) -> int:
        """
        Delete expired audio files.

        Returns:
            Number of deleted files
        """
        try:
            collection = await self._get_collection()

            result = await collection.delete_many({
                "expires_at": {"$lt": datetime.utcnow()}
            })

            if result.deleted_count > 0:
                logger.info(f"Cleaned up {result.deleted_count} expired audio files")

            return result.deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up expired audio files: {str(e)}")
            return 0

    async def get_by_conversation_id(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get audio files for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            List of audio files (without content)
        """
        return await self.list(filters={"conversation_id": conversation_id})
