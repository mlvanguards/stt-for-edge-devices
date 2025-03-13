import logging
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB connection manager with serverless-friendly reconnection logic"""

    client = None
    db = None
    conversations_collection = None
    messages_collection = None
    _initialized = False

    @classmethod
    async def connect(cls):
        """Connect to MongoDB and initialize collections and indexes"""
        from src.config.settings import (
            MONGODB_URI,
            MONGODB_DB,
            MONGODB_CONVERSATIONS_COLLECTION,
            MONGODB_MESSAGES_COLLECTION
        )

        if cls._initialized and cls.client is not None and cls.db is not None:
            # Already connected
            return True

        try:
            # Connect to MongoDB with connection pooling settings
            cls.client = AsyncIOMotorClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                retryWrites=True
            )

            # Test the connection
            await cls.client.admin.command('ping')

            cls.db = cls.client[MONGODB_DB]
            cls.conversations_collection = cls.db[MONGODB_CONVERSATIONS_COLLECTION]
            cls.messages_collection = cls.db[MONGODB_MESSAGES_COLLECTION]

            # Create indexes for better query performance
            await cls.conversations_collection.create_index("conversation_id", unique=True)
            await cls.messages_collection.create_index("conversation_id")
            await cls.messages_collection.create_index([("conversation_id", 1), ("timestamp", 1)])

            cls._initialized = True
            logger.info(f"Connected to MongoDB database: {MONGODB_DB}")
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
            cls._initialized = False
            logger.info("MongoDB connection closed")

    @classmethod
    async def ensure_connection(cls):
        """Ensure we have a valid connection, reconnecting if necessary"""
        if not cls._initialized or cls.client is None or cls.db is None:
            await cls.connect()
        return cls._initialized

    @classmethod
    async def get_conversation(cls, conversation_id):
        """Get a conversation by ID"""
        await cls.ensure_connection()
        return await cls.conversations_collection.find_one({"conversation_id": conversation_id})

    @classmethod
    async def get_conversation_messages(cls, conversation_id):
        """Get all messages for a conversation ID"""
        await cls.ensure_connection()
        cursor = cls.messages_collection.find({"conversation_id": conversation_id}).sort("timestamp", 1)
        return await cursor.to_list(length=None)

    @classmethod
    async def add_message(cls, conversation_id, role, content):
        """Add a message to a conversation"""
        from datetime import datetime
        await cls.ensure_connection()

        # Create message document
        message = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }

        # Insert the message
        await cls.messages_collection.insert_one(message)

        # Update the conversation's last_updated timestamp
        await cls.conversations_collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"last_updated": datetime.utcnow()}}
        )

    @classmethod
    async def create_conversation(cls, conversation_id, system_prompt, voice_id):
        """Create a new conversation"""
        from datetime import datetime
        await cls.ensure_connection()

        # Create conversation document
        conversation = {
            "conversation_id": conversation_id,
            "system_prompt": system_prompt,
            "voice_id": voice_id,
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow()
        }

        # Insert the conversation
        await cls.conversations_collection.insert_one(conversation)

        # Add system message
        await cls.add_message(conversation_id, "system", system_prompt)

    @classmethod
    async def delete_conversation(cls, conversation_id):
        """Delete a conversation and all its messages"""
        await cls.ensure_connection()

        # Delete the conversation
        result = await cls.conversations_collection.delete_one({"conversation_id": conversation_id})

        # Delete all messages for the conversation
        await cls.messages_collection.delete_many({"conversation_id": conversation_id})

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

        return {
            "total": total,
            "conversations": conversations,
            "page": skip // limit + 1 if limit > 0 else 1,
            "limit": limit,
            "pages": (total + limit - 1) // limit if limit > 0 else 1
        }