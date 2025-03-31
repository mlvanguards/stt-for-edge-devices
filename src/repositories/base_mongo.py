import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic

from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import PyMongoError

from src.core.interfaces.repository import IRepository
from src.database.mongo import mongodb_connection

T = TypeVar('T')

logger = logging.getLogger(__name__)


class BaseMongoRepository(Generic[T], IRepository[T]):
    """
    Base MongoDB repository implementation with shared functionality.

    This class provides common CRUD operations and error handling
    for MongoDB repositories. Concrete repositories should inherit from this
    class and specify the collection name.
    """

    _collection_name: str = None
    _collection: Optional[AsyncIOMotorCollection] = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """
        Get the MongoDB collection, ensuring connection is established.

        Returns:
            MongoDB collection for the repository
        """
        if self._collection is None:
            if not self._collection_name:
                raise ValueError("Collection name not specified for repository")

            await mongodb_connection.ensure_connection()
            self._collection = mongodb_connection.db[self._collection_name]

        return self._collection

    async def create(self, entity: Dict[str, Any]) -> Optional[T]:
        """
        Create a new entity.

        Args:
            entity: Dictionary with entity properties

        Returns:
            The created entity if successful, None otherwise
        """
        try:
            collection = await self._get_collection()

            # Add timestamps if not present
            now = datetime.utcnow().isoformat()
            if "created_at" not in entity:
                entity["created_at"] = now
            if "updated_at" not in entity and "last_updated" not in entity:
                entity["updated_at"] = now

            result = await collection.insert_one(entity)

            if result.acknowledged:
                # Add the ID to the entity dict for the return value
                entity["_id"] = str(result.inserted_id)
                return entity

            return None

        except PyMongoError as e:
            logger.error(f"MongoDB error creating entity: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating entity: {str(e)}")
            return None

    async def read(self, id: str) -> Optional[T]:
        """
        Read a entity by its ID.

        Args:
            id: The entity ID

        Returns:
            The entity if found, None otherwise
        """
        try:
            collection = await self._get_collection()

            # Determine the ID field (either _id or custom ID field)
            if ObjectId.is_valid(id):
                id_field = "_id"
                id_value = ObjectId(id)
            else:
                # Repositories should override for custom IDs
                id_field = self._get_id_field_name()
                id_value = id

            entity = await collection.find_one({id_field: id_value})

            # Process the entity if found
            if entity:
                entity = self._serialize_dates(entity)
                entity["_id"] = str(entity["_id"])

            return entity

        except PyMongoError as e:
            logger.error(f"MongoDB error reading entity: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading entity: {str(e)}")
            return None

    async def update(self, id: str, entity: Dict[str, Any]) -> bool:
        """
        Update an entity.

        Args:
            id: The entity identifier
            entity: Dictionary with updated properties

        Returns:
            True if update successful, False otherwise
        """
        try:
            collection = await self._get_collection()

            # Update timestamp
            entity["updated_at"] = datetime.utcnow().isoformat()

            # Determine the ID field (either _id or custom ID field)
            if ObjectId.is_valid(id):
                id_field = "_id"
                id_value = ObjectId(id)
            else:
                # Repositories should override for custom IDs
                id_field = self._get_id_field_name()
                id_value = id

            result = await collection.update_one(
                {id_field: id_value},
                {"$set": entity}
            )

            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(f"MongoDB error updating entity: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating entity: {str(e)}")
            return False

    async def delete(self, id: str) -> bool:
        """
        Delete an entity.

        Args:
            id: The entity identifier

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            collection = await self._get_collection()

            # Determine the ID field (either _id or custom ID field)
            if ObjectId.is_valid(id):
                id_field = "_id"
                id_value = ObjectId(id)
            else:
                # Repositories should override for custom IDs
                id_field = self._get_id_field_name()
                id_value = id

            result = await collection.delete_one({id_field: id_value})

            return result.deleted_count > 0

        except PyMongoError as e:
            logger.error(f"MongoDB error deleting entity: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting entity: {str(e)}")
            return False

    async def list(self, skip: int = 0, limit: int = 100, filters: Dict[str, Any] = None) -> List[T]:
        """
        List entities with pagination and optional filtering.

        Args:
            skip: Number of entities to skip
            limit: Maximum number of entities to return
            filters: Optional dictionary of filter criteria

        Returns:
            List of entities
        """
        try:
            collection = await self._get_collection()

            # Prepare filter criteria
            filter_criteria = filters or {}

            # Get entities with pagination
            cursor = collection.find(filter_criteria).skip(skip).limit(limit)
            entities = await cursor.to_list(length=limit)

            # Process entities
            result = []
            for entity in entities:
                entity = self._serialize_dates(entity)
                entity["_id"] = str(entity["_id"])
                result.append(entity)

            return result

        except PyMongoError as e:
            logger.error(f"MongoDB error listing entities: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing entities: {str(e)}")
            return []

    def _get_id_field_name(self) -> str:
        """
        Get the name of the ID field used by this repository.
        Subclasses can override this method for custom ID fields.

        Returns:
            Field name for the entity ID
        """
        return "_id"

    def _serialize_dates(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert datetime objects to ISO format strings for serialization.

        Args:
            entity: The entity to process

        Returns:
            Entity with serialized date fields
        """
        for key, value in entity.items():
            if isinstance(value, datetime):
                entity[key] = value.isoformat()

        return entity
