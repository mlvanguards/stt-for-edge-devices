import logging
from fastapi import APIRouter, status
from typing import Optional
from src.models.api import ApiKeySubmission, ApiKeyInfo, ApiKeyDetailsResponse

# Setup logger
logger = logging.getLogger(__name__)

API_KEYS = {
    "HUGGINGFACE_TOKEN": None,
    "OPENAI_API_KEY": None,
    "ELEVENLABS_API_KEY": None
}

# Description of each API key for better user guidance
API_KEY_DESCRIPTIONS = {
    "HUGGINGFACE_TOKEN": "Required for speech recognition",
    "OPENAI_API_KEY": "Required for chat functionality",
    "ELEVENLABS_API_KEY": "Required for text-to-speech",
}

# Create router
router = APIRouter(
    prefix="/api-keys",
    tags=["API Keys Management"],
    responses={404: {"description": "Not found"}}
)


# Functions to access API keys from anywhere in the application
def get_api_key(key_name: str) -> Optional[str]:
    """Get an API key by name"""
    return API_KEYS.get(key_name)


def are_all_keys_set() -> bool:
    """Check if all required API keys are set"""
    return all(API_KEYS.values())


def get_missing_keys() -> list:
    """Get list of missing API keys"""
    return [key for key, value in API_KEYS.items() if not value]


# Routes
@router.post("/submit", status_code=status.HTTP_200_OK)
async def submit_api_keys(keys: ApiKeySubmission):
    """
    Submit API keys for the application to use.

    These keys will be stored in memory and used for API requests instead of environment variables.
    """
    # Store the provided keys
    API_KEYS["HUGGINGFACE_TOKEN"] = keys.huggingface_token
    API_KEYS["OPENAI_API_KEY"] = keys.openai_api_key
    API_KEYS["ELEVENLABS_API_KEY"] = keys.elevenlabs_api_key

    logger.info("API keys updated by user")

    # Return current status
    return {
        "status": "success",
        "message": "API keys have been updated",
        "all_keys_set": are_all_keys_set()
    }


@router.get("/status")
async def get_api_key_status():
    """
    Get the current status of API keys.

    Returns which keys are set and which are missing.
    """
    all_set = are_all_keys_set()
    missing = get_missing_keys() if not all_set else []

    response = ApiKeyDetailsResponse(
        keys={
            key: ApiKeyInfo(
                key_name=key,
                description=API_KEY_DESCRIPTIONS.get(key, "Required API key"),
                is_set=bool(value)
            )
            for key, value in API_KEYS.items()
        },
        all_keys_set=all_set,
        missing_keys=missing
    )

    return response


@router.delete("/reset")
async def reset_api_keys():
    """
    Reset all stored API keys.

    This will clear all user-provided API keys from memory.
    """
    for key in API_KEYS:
        API_KEYS[key] = None

    logger.info("All API keys have been reset")

    return {
        "status": "success",
        "message": "All API keys have been reset"
    }
