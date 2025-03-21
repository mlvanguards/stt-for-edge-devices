import logging
import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB connection manager with enhanced memory support"""

    client = None
    db = None
    conversations_collection = None
    messages_collection = None
    memory_collection = None  # New collection for memory summaries
    _initialized = False

    @classmethod
    async def connect(cls):
        """Connect to MongoDB and initialize collections and indexes"""
        from src.config.settings import (
            MONGODB_CONVERSATIONS_COLLECTION,
            MONGODB_MESSAGES_COLLECTION
        )

        if cls._initialized and cls.client is not None and cls.db is not None:
            # Already connected
            return True

        try:
            # Get connection string from environment - use genezio's environment variable format
            mongodb_uri = os.getenv("MONGODB_URI")
            mongodb_db = os.getenv("MONGODB_DB", "stt-app-db")

            if not mongodb_uri:
                logger.error("No MongoDB connection string found in environment variables")
                return False

            # Connect to MongoDB with connection pooling settings optimized for serverless
            cls.client = AsyncIOMotorClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                retryWrites=True
            )

            # Test the connection
            await cls.client.admin.command('ping')

            # Use the database name from environment or fallback
            cls.db = cls.client[mongodb_db]
            cls.conversations_collection = cls.db[MONGODB_CONVERSATIONS_COLLECTION]
            cls.messages_collection = cls.db[MONGODB_MESSAGES_COLLECTION]
            cls.memory_collection = cls.db["memory_summaries"]  # New collection for memory

            # Create indexes for better query performance
            await cls.conversations_collection.create_index("conversation_id", unique=True)
            await cls.messages_collection.create_index("conversation_id")
            await cls.messages_collection.create_index([("conversation_id", 1), ("timestamp", 1)])

            # New indices for memory features
            await cls.messages_collection.create_index([("conversation_id", 1), ("importance", -1)])
            await cls.memory_collection.create_index("conversation_id")

            cls._initialized = True
            logger.info(f"Connected to MongoDB database: {mongodb_db}")
            return True
        except Exception as e:
            cls._initialized = False
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False

    @classmethod
    async def close(cls):
        """Close MongoDB connection"""
        if cls.client is not None:
            cls.client.close()
            cls.client = None
            cls.db = None
            cls.conversations_collection = None
            cls.messages_collection = None
            cls.memory_collection = None
            cls._initialized = False
            logger.info("MongoDB connection closed")

    @classmethod
    async def ensure_connection(cls):
        """Ensure we have a valid connection, reconnecting if necessary"""
        if not cls._initialized or cls.client is None or cls.db is None:
            await cls.connect()

        # Additional check for valid connection
        if cls._initialized and cls.client is not None and cls.db is not None:
            try:
                # Test the connection with a simple command
                await cls.client.admin.command('ping')
                return True
            except Exception as e:
                logger.error(f"Connection test failed: {str(e)}")
                cls._initialized = False
                await cls.connect()  # Try to reconnect

    @classmethod
    async def get_conversation(cls, conversation_id):
        """Get a conversation by ID"""
        await cls.ensure_connection()
        return await cls.conversations_collection.find_one({"conversation_id": conversation_id})

    @classmethod
    async def get_conversation_messages(cls, conversation_id):
        """
        Get all messages for a conversation ID with string timestamps
        """
        await cls.ensure_connection()
        cursor = cls.messages_collection.find({"conversation_id": conversation_id}).sort("timestamp", 1)
        messages = await cursor.to_list(length=None)

        # Convert datetime objects to string timestamps
        for message in messages:
            if "timestamp" in message and isinstance(message["timestamp"], datetime):
                message["timestamp"] = message["timestamp"].isoformat()

        return messages

    @classmethod
    async def add_message(cls, conversation_id, role, content, timestamp=None, importance=None):
        """
        Add a message to a conversation with enhanced metadata

        Args:
            conversation_id: ID of the conversation
            role: Message role (user, assistant, system)
            content: Message content
            timestamp: Optional timestamp (defaults to now)
            importance: Optional importance score (0-1)
        """
        await cls.ensure_connection()

        # Create message document with enhanced fields
        message = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() if timestamp is None else timestamp
        }

        # Add importance if provided (for memory prioritization)
        if importance is not None:
            message["importance"] = importance

        # Insert the message
        await cls.messages_collection.insert_one(message)

        # Update the conversation's last_updated timestamp
        await cls.conversations_collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"last_updated": datetime.utcnow().isoformat()}}
        )

        return message

    @classmethod
    async def create_conversation(cls, conversation_id, system_prompt, voice_id, stt_model_id=None):
        """Create a new conversation with string timestamps"""
        from src.config.settings import DEFAULT_STT_MODEL_ID

        now = datetime.utcnow().isoformat()
        await cls.ensure_connection()

        # Create conversation document with more metadata
        conversation = {
            "conversation_id": conversation_id,
            "system_prompt": system_prompt,
            "voice_id": voice_id,
            "stt_model_id": stt_model_id or DEFAULT_STT_MODEL_ID,
            "created_at": now,
            "last_updated": now,
            "message_count": 0,
            "memory_optimized": False  # Track if memory optimization has been applied
        }

        # Insert the conversation
        await cls.conversations_collection.insert_one(conversation)

        # Add system message with timestamp
        await cls.add_message(conversation_id, "system", system_prompt, timestamp=now)

    @classmethod
    async def update_conversation_summary(cls, conversation_id, summary_text):
        """
        Store or update a conversation summary for memory optimization

        Args:
            conversation_id: ID of the conversation
            summary_text: GPT-generated summary of the conversation
        """
        await cls.ensure_connection()

        # Check if summary exists
        existing = await cls.memory_collection.find_one({"conversation_id": conversation_id})

        timestamp = datetime.utcnow().isoformat()

        if existing:
            # Update existing summary
            await cls.memory_collection.update_one(
                {"conversation_id": conversation_id},
                {
                    "$set": {
                        "summary": summary_text,
                        "updated_at": timestamp
                    }
                }
            )
        else:
            # Create new summary
            await cls.memory_collection.insert_one({
                "conversation_id": conversation_id,
                "summary": summary_text,
                "created_at": timestamp,
                "updated_at": timestamp
            })

        # Mark conversation as memory-optimized
        await cls.conversations_collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"memory_optimized": True}}
        )

    @classmethod
    async def get_conversation_summary(cls, conversation_id):
        """Get the stored summary for a conversation"""
        await cls.ensure_connection()
        summary_doc = await cls.memory_collection.find_one({"conversation_id": conversation_id})
        return summary_doc["summary"] if summary_doc else None

    @classmethod
    async def delete_conversation(cls, conversation_id):
        """Delete a conversation and all its messages"""
        await cls.ensure_connection()

        # Delete the conversation
        result = await cls.conversations_collection.delete_one({"conversation_id": conversation_id})

        # Delete all messages for the conversation
        await cls.messages_collection.delete_many({"conversation_id": conversation_id})

        # Delete memory summaries
        await cls.memory_collection.delete_many({"conversation_id": conversation_id})

        return result.deleted_count > 0

    @classmethod
    async def list_conversations(cls, limit=10, skip=0):
        """List conversations with pagination"""
        await cls.ensure_connection()

        # Get total count
        total = await cls.conversations_collection.count_documents({})

        # Get conversations with pagination
        cursor = cls.conversations_collection.find().sort("last_updated", -1).skip(skip).limit(limit)
        conversations = await cursor.to_list(length=limit)

        # Convert datetime objects to ISO format strings for JSON serialization
        for conv in conversations:
            if "created_at" in conv and isinstance(conv["created_at"], datetime):
                conv["created_at"] = conv["created_at"].isoformat()
            if "last_updated" in conv and isinstance(conv["last_updated"], datetime):
                conv["last_updated"] = conv["last_updated"].isoformat()

        return {
            "total": total,
            "conversations": conversations,
            "page": skip // limit + 1 if limit > 0 else 1,
            "limit": limit,
            "pages": (total + limit - 1) // limit if limit > 0 else 1
        }

    @classmethod
    async def update_message_importance(cls, message_id, importance):
        """
        Update the importance score for a message (for memory prioritization)

        Args:
            message_id: ID of the message
            importance: Importance score (0-1)
        """
        await cls.ensure_connection()

        result = await cls.messages_collection.update_one(
            {"_id": message_id},
            {"$set": {"importance": importance}}
        )

        return result.modified_count > 0