import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.core.interfaces.repository import IAudioRepository
from src.core.interfaces.service import IExternalAPIClient
from src.core.utils.audio.audio_handling import AudioProcessor

logger = logging.getLogger(__name__)


class SpeechRecognitionService):
    """
    Service for handling speech recognition operations.
    Implements the ISpeechRecognitionService interface.
    Uses the centralized AudioProcessor for all audio processing needs.
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
            audio_processor: Audio processing utility
        """
        self.external_api_client = external_api_client
        self.audio_repository = audio_repository
        self.audio_processor = audio_processor or AudioProcessor()
        self.default_model_id = settings.stt.DEFAULT_STT_MODEL_ID

    async def process_audio_file(
            self,
            audio_content: bytes,
            content_type: str,
            model_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            store_audio: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process an audio file by sending it to the Hugging Face API.

        Args:
            audio_content: Raw audio bytes
            content_type: MIME type of the audio
            model_id: Optional model ID to use (defaults to default from settings)
            conversation_id: Optional conversation ID to associate with
            store_audio: Whether to store the audio in the repository

        Returns:
            List of transcription segments with text and metadata
        """
        try:
            # Use the selected model or fall back to the default
            selected_model = model_id if model_id else self.default_model_id
            logger.info(f"Processing audio with model {selected_model}")

            # Validate content type
            if not self.audio_processor.validate_content_type(content_type):
                logger.error(f"Invalid content type: {content_type}")
                return [{"index": 0, "text": "Error: Invalid audio format."}]

            # Optimize audio for speech recognition using the audio processor
            optimized_audio = self.audio_processor.optimize_for_stt(audio_content, content_type)
            if not optimized_audio:
                logger.error("Failed to optimize audio")
                return [{"index": 0, "text": "Error: Failed to process audio file."}]

            # Store the optimized audio if requested
            audio_id = None
            if store_audio:
                audio_id = await self.audio_repository.save_audio(
                    audio_content=optimized_audio,
                    content_type="audio/wav",  # Always WAV after optimization
                    conversation_id=conversation_id,
                    ttl_hours=24  # Default TTL
                )

            # Call Hugging Face API through the external client
            result = await self.external_api_client.call_huggingface_api(
                model_id=selected_model,
                audio_content=optimized_audio,
                content_type="audio/wav"  # Always WAV after optimization
            )

            if not result["success"]:
                logger.error(f"API error: {result.get('error')}")
                return [{
                    "index": 0,
                    "text": result["text"],
                    "audio_id": audio_id
                }]

            # Return the transcription as a segment
            return [{
                "index": 0,
                "text": result["text"],
                "audio_id": audio_id
            }]

        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {str(e)}")
            return [{"index": 0, "text": f"Error processing audio file: {str(e)}"}]

    def clean_transcription(self, transcriptions: List[Dict[str, Any]]) -> str:
        """
        Process and clean transcription results.

        Args:
            transcriptions: List of transcription segments

        Returns:
            Cleaned transcription text
        """
        # Sort transcriptions by index
        sorted_transcriptions = sorted(transcriptions, key=lambda x: x.get("index", 0))

        # Combine all transcriptions
        full_text = " ".join([t.get("text", "") for t in sorted_transcriptions])

        # Clean up the transcription text
        clean_text = re.sub(
            r"\[Segment \d+ transcription failed\]\s*", "", full_text
        ).strip()
        clean_text = re.sub(r"Failed to transcribe audio.*$", "", clean_text).strip()
        clean_text = re.sub(r"Error processing audio.*$", "", clean_text).strip()

        # If nothing is left after cleaning, return a default message
        if not clean_text:
            clean_text = "Unable to transcribe audio clearly. Please try again with a clearer recording."

        return clean_text

    async def get_audio_by_id(self, audio_id: str) -> tuple[Optional[bytes], Optional[str]]:
        """
        Retrieve previously processed audio by ID.

        Args:
            audio_id: The audio file ID

        Returns:
            Tuple of (audio_content, content_type) if found, (None, None) otherwise
        """
        return await self.audio_repository.get_audio(audio_id)

    async def warm_up_inference_api(self) -> None:
        """
        Send a small dummy request to the Hugging Face API to trigger model loading.
        Runs in the background during application startup.
        """
        try:
            logger.info(f"Warming up Hugging Face Inference API for model: {self.default_model_id}")

            # Generate a tiny audio file (0.5 seconds of silence) using audio processor
            audio_content = self.audio_processor.create_silent_audio(duration=0.5)

            # Send the request in background
            await self.external_api_client.call_huggingface_api(
                model_id=self.default_model_id,
                audio_content=audio_content,
                content_type="audio/wav"
            )

            # Wait a moment for model to initialize fully
            logger.info("Waiting for model to initialize...")
            await asyncio.sleep(2)
            logger.info("Warm-up complete")

        except Exception as e:
            logger.warning(f"Failed to warm up inference API: {str(e)}")
