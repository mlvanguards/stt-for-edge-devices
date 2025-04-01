import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None

    async def get_db(self) -> AsyncIOMotorDatabase:
        return self._client[settings.db.MONGODB_DB]

    def init(self):
        try:
            self._client = AsyncIOMotorClient(
                settings.db.MONGODB_URI,
                uuidRepresentation="standard",
                serverSelectionTimeoutMS=60000,
            )

            logger.info('Connected to mongo.')
        except Exception as e:
            logger.exception(f'Could not connect to mongo: {e}')
            raise

    async def close(self):
        if self._client:
            self._client.close()
            logger.info('Connection was closed.')

    async def connected(self):
        try:
            self._client.admin.command('ping')
        except Exception:
            return False
        return True


manager = DatabaseConnectionManager()
