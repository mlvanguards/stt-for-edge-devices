from typing import Optional
import uuid
from datetime import datetime

from pydantic import UUID4, ConfigDict, Field, BaseModel as _BaseModel


class BaseModel(_BaseModel):
    id: Optional[UUID4] = Field(default_factory=uuid.uuid4)

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        extra='ignore',
    )

    @classmethod
    def from_mongo(cls, data: dict):
        """Convert MongoDB document to Pydantic model."""
        if not data:
            return None

        mongo_id = data.pop("_id", None)

        if "id" not in data:
            data["id"] = uuid.uuid4()

        return cls(**data)

    def to_mongo(self, **kwargs) -> dict:
        """Convert model to MongoDB document format."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.model_dump(
            exclude_unset=exclude_unset, by_alias=by_alias, **kwargs
        )

        # Store conversation_id as the primary key instead of id
        if "conversation_id" in parsed:
            parsed["_id"] = parsed["conversation_id"]
        elif "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        return parsed


class TimestampMixin:
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
