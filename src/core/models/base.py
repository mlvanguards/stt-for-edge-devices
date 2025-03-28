from typing import Annotated, Optional
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator
from bson import ObjectId


def validate_object_id(v) -> ObjectId:
    """Validate and convert to ObjectId"""
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str) and ObjectId.is_valid(v):
        return ObjectId(v)
    raise ValueError("Invalid ObjectId")


# Type for ObjectId fields
PyObjectId = Annotated[ObjectId, BeforeValidator(validate_object_id)]


class MongoBaseModel(BaseModel):
    """Base model for MongoDB documents"""
    id: Optional[PyObjectId] = Field(alias="_id")

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )
