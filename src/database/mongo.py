import logging
from typing import Optional
import asyncio
import time
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.config.settings import settings

logger = logging.getLogger(__name__)


class MongoDBConnection:
    """
    MongoDB connection manager with connection pooling optimized for serverless.

    Implements a resilient connection strategy with reconnect capabilities
    and connection reuse across serverless function invocations.
    """

    _instance = None  # Singleton instance
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None
    _initialized: bool = False
    _last_used: float = 0
    _reconnect_lock = asyncio.Lock()

    def __new__(cls):
        """Implement singleton pattern for the connection manager."""
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
        return cls._instance

    async def connect(self) -> bool:
        """
        Connect to MongoDB and initialize the database instance.

        Returns:
            True if connection successful, False otherwise
        """
        # Use lock to prevent multiple reconnect attempts
        async with self._reconnect_lock:
            if self._initialized and self._client is not None and self._db is not None:
                # Already connected, update last used time
                self._last_used = time.time()
                return True

            try:
                # Get connection string from environment
                mongodb_uri = settings.db.MONGODB_URI
                mongodb_db = settings.db.MONGODB_DB

                if not mongodb_uri:
                    logger.error("No MongoDB connection string found in environment variables")
                    return False

                # Connect to MongoDB with connection pooling settings optimized for serverless
                # Using min pool size of 1 to keep at least one connection alive
                self._client = AsyncIOMotorClient(
                    mongodb_uri,
                    serverSelectionTimeoutMS=settings.db.MONGODB_SERVER_SELECTION_TIMEOUT_MS,
                    connectTimeoutMS=settings.db.MONGODB_CONNECT_TIMEOUT_MS,
                    maxPoolSize=settings.db.MONGODB_MAX_POOL_SIZE,
                    minPoolSize=settings.db.MONGODB_MIN_POOL_SIZE,
                    maxIdleTimeMS=settings.db.MONGODB_MAX_IDLE_TIME_MS,
                    retryWrites=settings.db.MONGODB_RETRY_WRITES
                )

                # Test the connection
                await self._client.admin.command('ping')

                # Use the database name from environment or fallback
                self._db = self._client[mongodb_db]

                self._initialized = True
                self._last_used = time.time()
                logger.info(f"Connected to MongoDB database: {mongodb_db}")
                return True
            except Exception as e:
                self._initialized = False
                logger.error(f"Failed to connect to MongoDB: {str(e)}")
                return False

    async def close(self) -> None:
        """Close MongoDB connection."""
        async with self._reconnect_lock:
            if self._client is not None:
                self._client.close()
                self._client = None
                self._db = None
                self._initialized = False
                logger.info("MongoDB connection closed")

    async def ensure_connection(self) -> bool:
        """
        Ensure we have a valid connection, reconnecting if necessary.

        This method will:
        1. Try to reuse an existing connection if it exists and is recent
        2. Test the connection with a lightweight operation
        3. Reconnect if the connection is stale or has failed

        Returns:
            True if connection is valid, False otherwise
        """
        # If never connected or explicitly closed, connect
        if not self._initialized or self._client is None or self._db is None:
            return await self.connect()

        # Check if connection is stale (unused for more than 10 minutes)
        if time.time() - self._last_used > 600:  # 10 minutes
            logger.info("Connection possibly stale, testing...")

            # Test with a connection-only command that doesn't hit the database
            try:
                await self._client.admin.command('ping')
                self._last_used = time.time()
                return True
            except Exception as e:
                logger.warning(f"Stale connection test failed: {str(e)}")
                await self.close()  # Explicitly close before reconnecting
                return await self.connect()

        # Update last used time
        self._last_used = time.time()
        return True

    @property
    def db(self) -> AsyncIOMotorDatabase:
        """
        Get the database instance.

        Returns:
            MongoDB database instance

        Raises:
            RuntimeError: If not connected to the database
        """
        if not self._initialized or self._db is None:
            raise RuntimeError("Not connected to MongoDB. Call connect() first.")
        return self._db

    @property
    def client(self) -> AsyncIOMotorClient:
        """
        Get the MongoDB client instance.

        Returns:
            MongoDB client instance

        Raises:
            RuntimeError: If not connected to the database
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("Not connected to MongoDB. Call connect() first.")
        return self._client


# Singleton instance
mongodb_connection = MongoDBConnection()
