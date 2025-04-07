import time
import logging
from typing import Optional, Dict, Any
import requests

from src.config.settings import settings

logger = logging.getLogger(__name__)


class HuggingFaceGatewayClient:
    """Client for interacting with HuggingFace Inference API"""

    def __init__(self, base_url: str = "https://api-inference.huggingface.co/models"):
        self._base_url = base_url
        self._api_key = settings.auth.HUGGINGFACE_TOKEN

    def speech_to_text(
            self,
            model_id: str,
            audio_content: bytes,
            content_type: str,
            max_retries: Optional[int] = None,
            backoff_factor: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call Hugging Face Inference API for speech recognition.

        Args:
            model_id: HF model identifier
            audio_content: Audio bytes
            content_type: MIME type
            max_retries: Optional retry count
            backoff_factor: Optional backoff multiplier

        Returns:
            Response with transcription or error
        """
        # Use default settings if not provided
        _max_retries = max_retries or settings.stt.SPEECH_RECOGNITION_RETRIES
        _backoff_factor = backoff_factor or settings.stt.SPEECH_RECOGNITION_BACKOFF_FACTOR

        # Construct the full API URL with the selected model
        api_url = f"{self._base_url}/{model_id}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": content_type,
        }

        # Try multiple times with exponential backoff
        for attempt in range(_max_retries):
            try:
                logger.info(f"HF API attempt {attempt + 1}/{_max_retries} for model {model_id}")

                # Use a longer timeout for the first attempt (model loading)
                timeout = 30.0 if attempt == 0 else 15.0

                response = requests.post(
                    api_url,
                    headers=headers,
                    data=audio_content,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    try:
                        transcription_result = response.json()
                    except ValueError:
                        # Not JSON, treat as plain text
                        transcription_result = {"text": response.text}

                    # Extract the text from the transcription
                    if isinstance(transcription_result, dict) and "text" in transcription_result:
                        text = transcription_result["text"]
                    else:
                        text = str(transcription_result)

                    # Check for failure markers in the text
                    failure_markers = [
                        "Failed to transcribe",
                        "Error processing audio",
                        "failed to transcribe",
                    ]

                    if text and not any(marker in text for marker in failure_markers):
                        return {
                            "success": True,
                            "text": text
                        }
                    else:
                        logger.warning(f"API returned success but with failure message: {text}")
                        # This is likely a loading issue, so wait longer and retry
                        wait_time = max(3, (_backoff_factor ** attempt) * 2)
                        logger.info(f"Waiting {wait_time}s before retrying...")
                        time.sleep(wait_time)
                        continue

                elif response.status_code == 401:
                    # Authentication error - likely invalid token
                    logger.error("Authentication failed. Please provide a valid HuggingFace API key.")
                    return {
                        "success": False,
                        "error": "Invalid HuggingFace API key",
                        "text": "Error: Invalid HuggingFace API key. Please provide a valid API key."
                    }

                elif response.status_code == 503:
                    # Service unavailable, likely model loading
                    wait_time = max(5, (_backoff_factor ** attempt) * 2)
                    logger.warning(f"API returned 503, model likely loading. Retry in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Other error
                    logger.error(f"API error: {response.status_code} - {response.text}")

                    if attempt < _max_retries - 1:
                        # Wait longer before retrying
                        wait_time = (_backoff_factor ** attempt) * 1.5 + 1
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed
                        return {
                            "success": False,
                            "error": f"API error {response.status_code}",
                            "text": f"Failed to transcribe audio: API error {response.status_code}"
                        }

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                # For timeouts, use longer delays
                wait_time = (_backoff_factor ** attempt) * 2 + 3
                if attempt < _max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds after timeout...")
                    time.sleep(wait_time)
                else:
                    return {
                        "success": False,
                        "error": "Transcription timed out",
                        "text": "Transcription timed out"
                    }

            except Exception as e:
                logger.error(f"Error in processing attempt {attempt + 1}: {str(e)}")

                if attempt < _max_retries - 1:
                    # Try again with increased delay
                    wait_time = (_backoff_factor ** attempt) + 2
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Last attempt failed
                    return {
                        "success": False,
                        "error": str(e),
                        "text": f"Error processing audio: {str(e)}"
                    }

        # If we're here, all attempts failed
        return {
            "success": False,
            "error": "Failed after multiple attempts",
            "text": "Failed to transcribe audio after multiple attempts"
        }

    def warm_up_inference_api(self, model_id: Optional[str] = None, audio_content: Optional[bytes] = None) -> None:
        """
        Send a small dummy request to the Hugging Face API to trigger model loading.

        Args:
            model_id: The model ID to warm up
            audio_content: Small audio content for warm-up
        """
        # Use default model ID if not provided
        _model_id = model_id or settings.stt.DEFAULT_STT_MODEL_ID

        # Create silent audio if not provided
        if audio_content is None:
            # Create a simple silent audio bytes object
            from src.utils.audio.audio_handling import AudioProcessorMainApp
            audio_processor = AudioProcessorMainApp()
            audio_content = audio_processor.create_silent_audio(duration=0.5)

        logger.info(f"Warming up Hugging Face Inference API for model: {_model_id}")

        # Use a lower retry count for warm-up
        self.speech_to_text(
            model_id=_model_id,
            audio_content=audio_content,
            content_type="audio/wav",
            max_retries=1
        )
