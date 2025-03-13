import logging
from bson.binary import Binary
from datetime import datetime, timedelta
from bson.objectid import ObjectId

logger = logging.getLogger(__name__)


class AudioStorage:
    """
    Handles audio storage and retrieval using MongoDB
    without requiring ffmpeg or complex audio processing
    """

    @staticmethod
    async def save_audio(db, audio_content, content_type, conversation_id=None, ttl_hours=24):
        """
        Save audio content to MongoDB

        Args:
            db: MongoDB database instance
            audio_content: Raw audio bytes
            content_type: MIME type of the audio
            conversation_id: Optional conversation ID to associate with
            ttl_hours: Time-to-live in hours for automatic cleanup

        Returns:
            audio_id: ID of the stored audio
        """
        try:
            # Create audio document
            audio_doc = {
                "content": Binary(audio_content),  # Store as Binary BSON type
                "content_type": content_type,
                "size_bytes": len(audio_content),
                "conversation_id": conversation_id,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(hours=ttl_hours)
            }

            # Insert into database
            result = await db.audio_files.insert_one(audio_doc)
            audio_id = str(result.inserted_id)

            logger.info(f"Saved audio file {audio_id} ({len(audio_content)} bytes)")
            return audio_id

        except Exception as e:
            logger.error(f"Error saving audio to MongoDB: {str(e)}")
            raise

    @staticmethod
    async def get_audio(db, audio_id):
        """
        Retrieve audio content from MongoDB

        Args:
            db: MongoDB database instance
            audio_id: ID of the stored audio

        Returns:
            (audio_content, content_type): Tuple of audio bytes and content type
        """
        try:
            # Find audio document
            audio_doc = await db.audio_files.find_one({"_id": ObjectId(audio_id)})

            if not audio_doc:
                logger.warning(f"Audio file {audio_id} not found")
                return None, None

            # Return audio content and type
            return audio_doc["content"], audio_doc["content_type"]

        except Exception as e:
            logger.error(f"Error retrieving audio from MongoDB: {str(e)}")
            return None, None

    @staticmethod
    async def cleanup_expired_audio(db):
        """Delete expired audio files from MongoDB"""
        try:
            result = await db.audio_files.delete_many({
                "expires_at": {"$lt": datetime.utcnow()}
            })

            if result.deleted_count > 0:
                logger.info(f"Cleaned up {result.deleted_count} expired audio files")

            return result.deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up expired audio files: {str(e)}")
            return 0