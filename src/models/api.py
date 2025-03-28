from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ApiKeySubmission(BaseModel):
    """Model for API key submission"""
    huggingface_token: str = Field(
        ..., description="HuggingFace API token for speech recognition"
    )
    openai_api_key: str = Field(
        ..., description="OpenAI API key for chat functionality"
    )
    elevenlabs_api_key: str = Field(
        ..., description="ElevenLabs API key for text-to-speech"
    )


class ApiKeyInfo(BaseModel):
    """Model for API key information"""
    key_name: str
    description: str
    is_set: bool


class ApiKeyDetailsResponse(BaseModel):
    """Detailed response with key information"""
    keys: Dict[str, ApiKeyInfo]
    all_keys_set: bool
    missing_keys: Optional[List[str]] = None
