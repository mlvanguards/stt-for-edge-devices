import uuid
import base64
import logging
import requests
from typing import Optional, Tuple

from src.config.settings import settings
from src.utils.api_keys_service import get_elevenlabs_api_key

logger = logging.getLogger(__name__)


def synthesize_speech(text: str, voice_id: str = None) -> Tuple[bool, str, Optional[str]]:
    """
    Convert text to speech using ElevenLabs API.
    Requires user-provided API key.
    Returns (success, message, optional base64 encoded audio)
    """
    # Use default voice ID if none provided
    if voice_id is None:
        voice_id = settings.DEFAULT_VOICE_ID

    # Get the API key (user-provided only)
    elevenlabs_api_key = get_elevenlabs_api_key()

    if not elevenlabs_api_key:
        logger.error("ElevenLabs API key not available. User must provide an API key.")
        return False, "ElevenLabs API key is missing. Please provide your API key at /api-keys/elevenlabs", None

    if not text:
        logger.error("No text provided for speech synthesis")
        return False, "No text provided for speech synthesis", None

    headers = {
        "xi-api-key": elevenlabs_api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": settings.TTS_MODEL_ID,
        "voice_settings": settings.TTS_DEFAULT_SETTINGS
    }

    url = f"{settings.ELEVENLABS_API_URL}/{voice_id}"

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 401:
            logger.error("Authentication failed with ElevenLabs API. Please check your API key.")
            return False, "Invalid ElevenLabs API key. Please provide a valid API key at /api-keys/elevenlabs", None

        response.raise_for_status()

        # Generate a unique identifier for the audio
        file_id = str(uuid.uuid4())

        # Convert to base64 for API response
        audio_base64 = base64.b64encode(response.content).decode('utf-8')

        # Return a virtual path identifier and the base64 content
        return True, f"audio:{file_id}.mp3", audio_base64

    except requests.exceptions.RequestException as e:
        error_message = f"Error generating speech with ElevenLabs: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f" Response: {e.response.text}"
        logger.error(error_message)
        return False, error_message, None
    except Exception as e:
        error_message = f"Error generating speech: {str(e)}"
        logger.error(error_message)
        return False, error_message, None


async def get_available_voices():
    """
    Get a list of available voices from ElevenLabs.
    Requires user-provided API key.
    """
    # Get the API key (user-provided only)
    elevenlabs_api_key = get_elevenlabs_api_key()

    if not elevenlabs_api_key:
        logger.error("ElevenLabs API key not available. User must provide an API key.")
        return {
            "success": False,
            "error": "ElevenLabs API key is missing. Please provide your API key at /api-keys/elevenlabs"
        }

    headers = {
        "xi-api-key": elevenlabs_api_key
    }

    try:
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)

        if response.status_code == 401:
            logger.error("Authentication failed with ElevenLabs API. Please check your API key.")
            return {
                "success": False,
                "error": "Invalid ElevenLabs API key. Please provide a valid API key at /api-keys/elevenlabs"
            }

        response.raise_for_status()

        voices_data = response.json()

        # Format the voice data for the API response
        voices = [
            {
                "voice_id": voice["voice_id"],
                "name": voice["name"],
                "preview_url": voice.get("preview_url", None),
                "category": voice.get("category", "premium")
            }
            for voice in voices_data.get("voices", [])
        ]

        return {"success": True, "voices": voices}
    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching voices from ElevenLabs: {str(e)}"
        logger.error(error_message)
        return {"success": False, "error": error_message}
