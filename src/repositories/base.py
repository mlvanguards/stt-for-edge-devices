import uuid
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument, errors
from typing import Dict, Any, List, Tuple, Optional, Coroutine

from src.models.base import BaseModel
from src.erros import ImproperlyConfigured


class BaseRepository:

    model: BaseModel

    def __init__(self, db: AsyncIOMotorDatabase):
        if not hasattr(self.model, "Meta") or not hasattr(self.model.Meta, "name"):
            raise ImproperlyConfigured(
                "Document should define an Settings configuration class with the name of the collection."
            )

        self._collection = db[self.model.Settings.name]

    async def get(self, **filter_options) -> BaseModel:
        result = await self._collection.find_one(filter_options)
        return self.model.from_mongo(result)

    async def create(self, data: BaseModel) -> Dict[str, Any]:
        instance = await self._collection.insert_one(data.to_mongo())
        result = await self._collection.find_one({"_id": instance.inserted_id})
        return self.model.from_mongo(result)
    
    async def filter(
            self,
            projection: Dict[str, Any] = None,
            sort_by: List[tuple] = None,
            **filter_options
    ) -> List[BaseModel]:
        result = []

        cursor = self._collection.find(filter_options, projection=projection)

        if sort_by:
            cursor.sort(sort_by)

        async for document in cursor:
            result.append(self.model.from_mongo(document))

        return result
    
    async def list(self) -> List[BaseModel]:
        cursor = self._collection.find({})
        results = await cursor.to_list(length=None)
        return [self.model.from_mongo(result) for result in results]

    async def update(self, _id: uuid.UUID, data: Dict[str, Any], **kwargs) -> Optional[BaseModel]:
        instance = await self._collection.find_one_and_update(
            {"_id": _id},
            {"$set": data},
            upsert=False,
            return_document=ReturnDocument.AFTER
        )
        return self.model.from_mongo(instance) if instance else None
    
    async def delete_many(self, **filter_options) -> None:
        if not filter_options:
            raise ValueError("At least one filter must be provided for deletion")

        await self._collection.delete_many(filter_options)
        return None

    async def delete(self, _id: uuid.UUID) -> bool:
        result = await self._collection.delete_one({"_id": _id})
        return result.deleted_count > 0
