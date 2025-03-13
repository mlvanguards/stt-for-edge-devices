import uuid
import base64
import logging
import requests
from typing import Optional, Tuple

from src.config.settings import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_API_URL,
    DEFAULT_VOICE_ID,
    TTS_MODEL_ID,
    TTS_DEFAULT_SETTINGS
)

logger = logging.getLogger(__name__)


def synthesize_speech(text: str, voice_id: str = DEFAULT_VOICE_ID) -> Tuple[bool, str, Optional[str]]:
    """
    Convert text to speech using ElevenLabs API
    Returns (success, message, optional base64 encoded audio)
    """
    if not ELEVENLABS_API_KEY:
        logger.error("ElevenLabs API key not configured")
        return False, "ElevenLabs API key not configured", None

    if not text:
        logger.error("No text provided for speech synthesis")
        return False, "No text provided for speech synthesis", None

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": TTS_MODEL_ID,
        "voice_settings": TTS_DEFAULT_SETTINGS
    }

    url = f"{ELEVENLABS_API_URL}/{voice_id}"

    try:
        response = requests.post(url, json=payload, headers=headers)
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
    Get a list of available voices from ElevenLabs
    """
    if not ELEVENLABS_API_KEY:
        logger.error("ElevenLabs API key not configured")
        return {"success": False, "error": "ElevenLabs API key not configured"}

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }

    try:
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers)
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