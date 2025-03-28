import logging
from typing import Optional

from src.api.routes.api_keys import get_api_key

logger = logging.getLogger(__name__)


def get_effective_api_key(key_name: str) -> Optional[str]:
    """
    Get the effective API key to use for a service.

    Only returns user-provided keys, no fallback to environment variables.

    Args:
        key_name: The name of the API key to retrieve

    Returns:
        The API key value or None if not available
    """
    # Get the key from user-provided storage only
    user_key = get_api_key(key_name)

    if not user_key:
        logger.warning(f"API key {key_name} not found in user storage")

    return user_key


def check_api_key_availability(key_name: str) -> bool:
    """
    Check if an API key is available.

    Args:
        key_name: The name of the API key to check

    Returns:
        True if the key is available, False otherwise
    """
    return bool(get_effective_api_key(key_name))


def get_huggingface_token() -> Optional[str]:
    """Get the HuggingFace token"""
    return get_effective_api_key("HUGGINGFACE_TOKEN")


def get_openai_api_key() -> Optional[str]:
    """Get the OpenAI API key"""
    return get_effective_api_key("OPENAI_API_KEY")


def get_elevenlabs_api_key() -> Optional[str]:
    """Get the ElevenLabs API key"""
    return get_effective_api_key("ELEVENLABS_API_KEY")
