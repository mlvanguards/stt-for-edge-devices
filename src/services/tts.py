import logging
import base64
from typing import Dict, Optional, Any, Tuple

from src.config.settings import settings
from src.core.interfaces.service import ITextToSpeechService
from src.core.interfaces.repository import IAudioRepository
from src.core.interfaces.service import IExternalAPIClient
from src.core.utils.audio.audio_handling import AudioProcessor

logger = logging.getLogger(__name__)


class TextToSpeechService(ITextToSpeechService):
    """
    Service for handling text-to-speech operations.
    Implements the ITextToSpeechService interface.
    Uses the centralized AudioProcessor for audio processing needs.
    """

    def __init__(
            self,
            external_api_client: IExternalAPIClient,
            audio_repository: IAudioRepository,
            audio_processor: Optional[AudioProcessor] = None
    ):
        """
        Initialize with dependencies.

        Args:
            external_api_client: Client for API interactions
            audio_repository: Repository for audio storage
            audio_processor: Optional AudioProcessor instance
        """
        self.external_api_client = external_api_client
        self.audio_repository = audio_repository
        self.audio_processor = audio_processor or AudioProcessor()
        self.default_voice_id = settings.tts.DEFAULT_VOICE_ID
        self.tts_model_id = settings.tts.TTS_MODEL_ID
        self.voice_cache = None
        self.voice_cache_timestamp = None

    async def synthesize_speech(
            self, text: str, voice_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            return_base64: bool = True
    ) -> Dict[str, Any]:
        """
        Convert text to speech using ElevenLabs API and store the result.

        Args:
            text: The text to convert to speech
            voice_id: Optional voice ID to use
            conversation_id: Optional conversation ID to associate with
            return_base64: Whether to return the audio as base64 encoded string

        Returns:
            Dict with success status, file ID, audio data, and any error message
        """
        # Use default voice ID if none provided
        if voice_id is None:
            voice_id = self.default_voice_id

        # Call ElevenLabs API through the external client
        result = await self.external_api_client.call_elevenlabs_api(
            text=text,
            voice_id=voice_id,
            model_id=self.tts_model_id
        )

        if not result["success"]:
            logger.error(f"Failed to generate speech: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown error in speech synthesis"),
                "audio_base64": None,
                "file_id": None
            }

        try:
            # Get audio duration if possible
            audio_duration = None
            try:
                audio_content = result["audio_content"]
                audio_duration = self.audio_processor.get_audio_duration(audio_content)
                logger.info(f"Generated speech audio with duration: {audio_duration:.2f}s")
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {str(e)}")

            # Store audio in repository
            file_id = await self.audio_repository.save_audio(
                audio_content=result["audio_content"],
                content_type="audio/mpeg",  # ElevenLabs returns MP3
                conversation_id=conversation_id,
                ttl_hours=24  # Default TTL
            )

            # Return result
            response = {
                "success": True,
                "file_id": file_id,
                "error": None,
                "duration": audio_duration
            }

            # Add base64 encoded audio if requested
            if return_base64:
                audio_base64 = base64.b64encode(result["audio_content"]).decode("utf-8")
                response["audio_base64"] = audio_base64

            return response

        except Exception as e:
            logger.error(f"Error storing synthesized speech: {str(e)}")
            # Still return the audio even if storage failed
            if return_base64:
                audio_base64 = base64.b64encode(result["audio_content"]).decode("utf-8")
                return {
                    "success": True,
                    "audio_base64": audio_base64,
                    "file_id": None,
                    "error": f"Storage error: {str(e)}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Storage error: {str(e)}",
                    "audio_base64": None,
                    "file_id": None
                }

    async def get_audio_by_id(self, file_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Retrieve previously synthesized audio by ID.

        Args:
            file_id: The audio file ID

        Returns:
            Tuple of (audio_content, content_type) if found, (None, None) otherwise
        """
        return await self.audio_repository.get_audio(file_id)

    async def get_available_voices(self) -> Dict[str, Any]:
        """
        Get a list of available voices from ElevenLabs.

        Returns:
            Dict with success status and voices list or error message
        """
        # Check cache first (could add a timestamp check for freshness)
        if self.voice_cache is not None:
            return self.voice_cache

        # Call ElevenLabs API
        result = await self.external_api_client.get_elevenlabs_voices()

        # Cache the result if successful
        if result.get("success", False):
            self.voice_cache = result

        return result
