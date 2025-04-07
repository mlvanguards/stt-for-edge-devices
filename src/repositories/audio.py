import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from bson.objectid import ObjectId
from bson.binary import Binary
from motor.motor_asyncio import AsyncIOMotorCollection

from src.utils.audio.audio_handling import AudioProcessorMainApp
from src.repositories.base import BaseRepository
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class AudioModel(BaseModel):
    class Meta:
        name = "audio_files"

class AudioRepository(BaseRepository):
    model = AudioModel

    def __init__(self, db, audio_processor: Optional[AudioProcessorMainApp] = None):
        self.audio_processor = audio_processor or AudioProcessorMainApp()
        super().__init__(db)

    async def _get_collection(self) -> AsyncIOMotorCollection:
        collection = self._collection

        # Create indexes on this collection
        await collection.create_index("conversation_id")
        await collection.create_index("expires_at", expireAfterSeconds=0)
        return collection

    async def save_audio(
            self,
            audio_content: bytes,
            content_type: str,
            conversation_id: Optional[str] = None,
            ttl_hours: int = 24
    ) -> Optional[str]:
        try:
            if not self.audio_processor.validate_content_type(content_type):
                logger.warning(f"Invalid content type: {content_type}")
                return None

            duration = None
            # Only try to get duration for WAV files - skip for MP3
            if content_type in ["audio/wav", "audio/x-wav"]:
                try:
                    duration = self.audio_processor.get_audio_duration(audio_content)
                except Exception as e:
                    logger.warning(f"Could not determine audio duration: {str(e)}")

            audio_doc = {
                "content": Binary(audio_content),
                "content_type": content_type,
                "size_bytes": len(audio_content),
                "duration": duration,
                "conversation_id": conversation_id,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(hours=ttl_hours)
            }

            collection = await self._get_collection()
            result = await collection.insert_one(audio_doc)

            if result.inserted_id:
                audio_id = str(result.inserted_id)
                logger.info(f"Saved audio file {audio_id} ({len(audio_content)} bytes)")
                return audio_id

            return None

        except Exception as e:
            logger.error(f"Error saving audio to MongoDB: {str(e)}")
            return None

    async def get_audio(self, audio_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        try:
            collection = await self._get_collection()
            try:
                object_id = ObjectId(audio_id)
            except Exception:
                logger.warning(f"Invalid ObjectId: {audio_id}")
                return None, None

            audio_doc = await collection.find_one({"_id": object_id})

            if not audio_doc:
                logger.warning(f"Audio file {audio_id} not found")
                return None, None

            return audio_doc["content"], audio_doc["content_type"]

        except Exception as e:
            logger.error(f"Error retrieving audio from MongoDB: {str(e)}")
            return None, None

    async def cleanup_expired(self) -> int:
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
        return await self.list(filters={"conversation_id": conversation_id})
