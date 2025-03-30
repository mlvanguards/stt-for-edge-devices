import asyncio
import logging
import re
import time
import wave
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SpeechRecognitionService:
    """Service class for handling speech recognition operations."""

    def __init__(self):
        self.api_base_url = settings.stt.HUGGINGFACE_API_URL
        self.default_model_id = settings.stt.DEFAULT_STT_MODEL_ID
        self.max_retries = settings.stt.SPEECH_RECOGNITION_RETRIES
        self.backoff_factor = settings.stt.SPEECH_RECOGNITION_BACKOFF_FACTOR
        self.sample_rate = settings.audio.AUDIO_SAMPLE_RATE

    def process_audio_file(
        self,
        audio_content: bytes,
        content_type: str,
        model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process an audio file by sending it directly to HuggingFace API.
        Requires user-provided API key.

        Args:
            audio_content: Raw audio bytes
            content_type: MIME type of the audio
            model_id: Optional model ID to use for transcription (defaults to DEFAULT_STT_MODEL_ID)
            force_split: Ignored parameter (kept for API compatibility)

        Returns:
            List of transcription segments (typically just one)
        """
        try:
            # Get the API token (only user-provided)
            huggingface_token = settings.auth.HUGGINGFACE_TOKEN

            if not huggingface_token:
                logger.error(
                    "HuggingFace token not available. User must provide an API key."
                )
                return [
                    {
                        "index": 0,
                        "text": "Error: HuggingFace API key is missing. Please provide your API key at /api-keys/huggingface",
                    }
                ]

            # Always use a valid model ID - fall back to the default if none provided
            selected_model = model_id if model_id else self.default_model_id

            # Construct the full API URL with the selected model
            api_url = f"{self.api_base_url}/{selected_model}"

            logger.info(
                f"Processing audio with model {selected_model} (content type: {content_type})"
            )

            headers = {
                "Authorization": f"Bearer {huggingface_token}",
                "Content-Type": content_type,
            }

            # Try multiple times with exponential backoff
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Attempt {attempt + 1}/{self.max_retries}")

                    # Use a longer timeout for the first attempt (model loading)
                    timeout = 30.0 if attempt == 0 else 15.0

                    response = requests.post(
                        api_url,
                        headers=headers,
                        data=audio_content,
                        timeout=timeout,  # Add explicit timeout
                    )

                    if response.status_code == 200:
                        try:
                            transcription_result = response.json()
                        except ValueError:
                            # Not JSON, treat as plain text
                            transcription_result = {"text": response.text}

                        # Extract the text from the transcription
                        if (
                            isinstance(transcription_result, dict)
                            and "text" in transcription_result
                        ):
                            text = transcription_result["text"]
                        else:
                            text = str(transcription_result)

                        # Check for failure markers in the text
                        failure_markers = [
                            "Failed to transcribe",
                            "Error processing audio",
                            "failed to transcribe",
                        ]

                        if text and not any(
                            marker in text for marker in failure_markers
                        ):
                            return [{"index": 0, "text": text}]
                        else:
                            logger.warning(
                                f"API returned success but with failure message: {text}"
                            )
                            # This is likely a loading issue, so wait longer and retry
                            wait_time = max(3, (self.backoff_factor**attempt) * 2)
                            logger.info(f"Waiting {wait_time}s before retrying...")
                            time.sleep(wait_time)
                            continue

                    elif response.status_code == 401:
                        # Authentication error - likely invalid token
                        logger.error(
                            "Authentication failed. Please provide a valid HuggingFace API key."
                        )
                        return [
                            {
                                "index": 0,
                                "text": "Error: Invalid HuggingFace API key. Please provide a valid API key at /api-keys/huggingface",
                            }
                        ]

                    elif response.status_code == 503:
                        # Service unavailable, likely model loading
                        wait_time = max(5, (self.backoff_factor**attempt) * 2)
                        logger.warning(
                            f"API returned 503, model likely loading. Retry in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue

                    elif response.status_code in [400, 403]:
                        # Format issues or forbidden
                        logger.error(
                            f"API returned {response.status_code}: {response.text}"
                        )

                        # If first attempt, retry anyway - sometimes this happens on cold start
                        if attempt == 0:
                            logger.info(
                                "First attempt returned error - retrying anyway..."
                            )
                            time.sleep(3)
                            continue
                        elif attempt < self.max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            # Last attempt failed
                            return [
                                {
                                    "index": 0,
                                    "text": f"Failed to transcribe audio: API error {response.status_code}",
                                }
                            ]
                    else:
                        # Other error
                        logger.error(
                            f"API error: {response.status_code} - {response.text}"
                        )

                        if attempt < self.max_retries - 1:
                            # Wait longer before retrying
                            wait_time = (self.backoff_factor**attempt) * 1.5 + 1
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            # Last attempt failed
                            return [
                                {
                                    "index": 0,
                                    "text": f"Failed to transcribe audio: API error {response.status_code}",
                                }
                            ]

                except requests.exceptions.Timeout:
                    logger.warning(f"Request timeout on attempt {attempt + 1}")
                    # For timeouts, use longer delays
                    wait_time = (self.backoff_factor**attempt) * 2 + 3
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying in {wait_time} seconds after timeout...")
                        time.sleep(wait_time)
                    else:
                        return [{"index": 0, "text": "Transcription timed out"}]

                except Exception as e:
                    logger.error(f"Error in processing attempt {attempt + 1}: {str(e)}")

                    if attempt < self.max_retries - 1:
                        # Try again with increased delay
                        wait_time = (self.backoff_factor**attempt) + 2
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed
                        return [
                            {"index": 0, "text": f"Error processing audio: {str(e)}"}
                        ]

            # If we're here, all attempts failed
            return [
                {
                    "index": 0,
                    "text": "Failed to transcribe audio after multiple attempts",
                }
            ]

        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {str(e)}")
            return [{"index": 0, "text": "Error processing audio file."}]

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

        # Clean up the transcription text if it contains failure messages
        clean_text = re.sub(
            r"\[Segment \d+ transcription failed\]\s*", "", full_text
        ).strip()
        clean_text = re.sub(r"Failed to transcribe audio.*$", "", clean_text).strip()
        clean_text = re.sub(r"Error processing audio.*$", "", clean_text).strip()

        # If nothing is left after cleaning, return a default message
        if not clean_text:
            clean_text = "Unable to transcribe audio clearly. Please try again with a clearer recording."

        return clean_text

    async def warm_up_inference_api(self) -> None:
        """
        Send a small dummy request to the Hugging Face API to trigger model loading.
        Only runs if a user-provided API key is available.
        """
        try:
            # Get the API token (user-provided only)
            huggingface_token = settings.auth.HUGGINGFACE_TOKEN

            if not huggingface_token:
                logger.warning("Cannot warm up HuggingFace API: No API token available")
                return

            logger.info(
                f"Warming up Hugging Face Inference API for model: {self.default_model_id}"
            )

            # Generate a tiny audio file (0.5 seconds of silence)
            sample_rate = self.sample_rate
            duration = 0.5
            samples = np.zeros(int(sample_rate * duration), dtype=np.int16)

            buffer = BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())

            audio_content = buffer.getvalue()

            # Send warm-up request
            api_url = f"{self.api_base_url}/{self.default_model_id}"
            headers = {
                "Authorization": f"Bearer {huggingface_token}",
                "Content-Type": "audio/wav",
            }

            logger.info("Sending warm-up request to Hugging Face API")
            try:
                response = requests.post(
                    api_url, headers=headers, data=audio_content, timeout=45.0
                )
                logger.info(
                    f"Warm-up request completed with status {response.status_code}"
                )
            except Exception as e:
                logger.warning(f"Warm-up request failed: {str(e)}")

            # Wait a moment for model to initialize fully
            logger.info("Waiting for model to initialize...")
            await asyncio.sleep(2)
            logger.info("Warm-up complete")

        except Exception as e:
            logger.warning(f"Failed to warm up inference API: {str(e)}")


speech_recognition_service = SpeechRecognitionService()


if __name__ == "__main__":
    import asyncio

    speech_recognition_service = SpeechRecognitionService()

    async def test_speech_recognition():
        print("Testing Speech Recognition Service")
        print("==================================")

        # Test warm-up
        print("\n1. Testing API warm-up...")
        await speech_recognition_service.warm_up_inference_api()

        # Test with a sample audio file if provided
        audio_path = "/Users/vesaalexandru/Workspaces/cube/stt-for-edge-devices/data/M18_05_01.wav"

        # Read the audio file
        with open(audio_path, "rb") as f:
            audio_content = f.read()

        # Determine content type based on file extension
        content_type = "audio/wav"
        if audio_path.lower().endswith(".mp3"):
            content_type = "audio/mpeg"
        elif audio_path.lower().endswith(".ogg"):
            content_type = "audio/ogg"
        elif audio_path.lower().endswith(".flac"):
            content_type = "audio/flac"

        print(f"Content type: {content_type}")

        # Process the audio
        print("Processing audio...")
        transcriptions = speech_recognition_service.process_audio_file(
            audio_content, content_type
        )

        print("\nTranscription results:")
        for t in transcriptions:
            print(f"Segment {t.get('index', 0)}: {t.get('text', '')}")

        # Clean the transcription
        clean_text = speech_recognition_service.clean_transcription(transcriptions)
        print(f"\nCleaned transcription: {clean_text}")

    # Run the test
    asyncio.run(test_speech_recognition())
