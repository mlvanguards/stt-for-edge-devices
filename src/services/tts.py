import base64
import logging
import uuid
from typing import Optional, Tuple

import requests

from src.config.settings import settings

logger = logging.getLogger(__name__)


class TextToSpeechService:
    """Service class for handling text-to-speech operations."""

    def __init__(self):
        self.api_url = settings.tts.ELEVENLABS_API_URL
        self.default_voice_id = settings.tts.DEFAULT_VOICE_ID
        self.tts_model_id = settings.tts.TTS_MODEL_ID
        self.default_settings = settings.tts.TTS_DEFAULT_SETTINGS

    def synthesize_speech(
        self, text: str, voice_id: str = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Convert text to speech using ElevenLabs API.
        Requires user-provided API key.

        Args:
            text: The text to convert to speech
            voice_id: Optional voice ID to use (defaults to system default)

        Returns:
            Tuple of (success, message, optional base64 encoded audio)
        """
        # Use default voice ID if none provided
        if voice_id is None:
            voice_id = self.default_voice_id

        # Get the API key (user-provided only)
        elevenlabs_api_key = settings.auth.ELEVENLABS_API_KEY

        if not elevenlabs_api_key:
            logger.error(
                "ElevenLabs API key not available. User must provide an API key."
            )
            return (
                False,
                "ElevenLabs API key is missing. Please provide your API key at /api-keys/elevenlabs",
                None,
            )

        if not text:
            logger.error("No text provided for speech synthesis")
            return False, "No text provided for speech synthesis", None

        headers = {"xi-api-key": elevenlabs_api_key, "Content-Type": "application/json"}

        payload = {
            "text": text,
            "model_id": self.tts_model_id,
            "voice_settings": self.default_settings,
        }

        url = f"{self.api_url}/{voice_id}"

        try:
            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 401:
                logger.error(
                    "Authentication failed with ElevenLabs API. Please check your API key."
                )
                return (
                    False,
                    "Invalid ElevenLabs API key. Please provide a valid API key at /api-keys/elevenlabs",
                    None,
                )

            response.raise_for_status()

            # Generate a unique identifier for the audio
            file_id = str(uuid.uuid4())

            # Convert to base64 for API response
            audio_base64 = base64.b64encode(response.content).decode("utf-8")

            # Return a virtual path identifier and the base64 content
            return True, f"audio:{file_id}.mp3", audio_base64

        except requests.exceptions.RequestException as e:
            error_message = f"Error generating speech with ElevenLabs: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                error_message += f" Response: {e.response.text}"
            logger.error(error_message)
            return False, error_message, None
        except Exception as e:
            error_message = f"Error generating speech: {str(e)}"
            logger.error(error_message)
            return False, error_message, None

    async def get_available_voices(self):
        """
        Get a list of available voices from ElevenLabs.
        Requires user-provided API key.

        Returns:
            Dictionary with success status and either voices list or error message
        """
        # Get the API key (user-provided only)
        elevenlabs_api_key = settings.auth.ELEVENLABS_API_KEY

        if not elevenlabs_api_key:
            logger.error(
                "ElevenLabs API key not available. User must provide an API key."
            )
            return {
                "success": False,
                "error": "ElevenLabs API key is missing. Please provide your API key at /api-keys/elevenlabs",
            }

        headers = {"xi-api-key": elevenlabs_api_key}

        try:
            response = requests.get(
                "https://api.elevenlabs.io/v1/voices", headers=headers
            )

            if response.status_code == 401:
                logger.error(
                    "Authentication failed with ElevenLabs API. Please check your API key."
                )
                return {
                    "success": False,
                    "error": "Invalid ElevenLabs API key. Please provide a valid API key at /api-keys/elevenlabs",
                }

            response.raise_for_status()

            voices_data = response.json()

            # Format the voice data for the API response
            voices = [
                {
                    "voice_id": voice["voice_id"],
                    "name": voice["name"],
                    "preview_url": voice.get("preview_url", None),
                    "category": voice.get("category", "premium"),
                }
                for voice in voices_data.get("voices", [])
            ]

            return {"success": True, "voices": voices}
        except requests.exceptions.RequestException as e:
            error_message = f"Error fetching voices from ElevenLabs: {str(e)}"
            logger.error(error_message)
            return {"success": False, "error": error_message}


# Create a singleton instance of the service
text_to_speech_service = TextToSpeechService()


# Provide backward compatibility functions that delegate to the service
def synthesize_speech(
    text: str, voice_id: str = None
) -> Tuple[bool, str, Optional[str]]:
    """Backward compatibility function that delegates to the service."""
    return text_to_speech_service.synthesize_speech(text, voice_id)


async def get_available_voices():
    """Backward compatibility function that delegates to the service."""
    return await text_to_speech_service.get_available_voices()


if __name__ == "__main__":
    import asyncio

    async def test_tts_service():
        print("Testing Text-to-Speech Service")
        print("==============================")

        # Test voice listing
        print("\n1. Testing voice listing...")
        voices_result = await text_to_speech_service.get_available_voices()

        if voices_result["success"]:
            print(f"Successfully retrieved {len(voices_result['voices'])} voices")
            print("First 3 voices:")
            for voice in voices_result["voices"][:3]:
                print(
                    f"  - {voice['name']} (ID: {voice['voice_id']}, Category: {voice['category']})"
                )
        else:
            print(
                f"Failed to retrieve voices: {voices_result.get('error', 'Unknown error')}"
            )

        # Test speech synthesis
        print("\n2. Testing speech synthesis...")
        test_text = "Hello, this is a test of the text to speech service."

        success, message, audio_base64 = text_to_speech_service.synthesize_speech(
            test_text
        )

        if success:
            print(f"Successfully synthesized speech: {message}")
            print(
                f"Audio data length: {len(audio_base64) if audio_base64 else 0} characters"
            )

            # Optionally save the audio to a file for testing
            if audio_base64:
                try:
                    audio_data = base64.b64decode(audio_base64)
                    with open("test_tts_output.mp3", "wb") as f:
                        f.write(audio_data)
                    print("Saved audio to test_tts_output.mp3")
                except Exception as e:
                    print(f"Error saving audio file: {str(e)}")
        else:
            print(f"Failed to synthesize speech: {message}")

    # Run the test
    asyncio.run(test_tts_service())
