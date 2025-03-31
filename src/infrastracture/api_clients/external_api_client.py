import logging
import time
import requests
from typing import Dict, Any, Optional

from src.config.settings import settings
from src.core.interfaces.service import IExternalAPIClient

logger = logging.getLogger(__name__)


class ExternalAPIClient(IExternalAPIClient):
    """Implementation of external API client interface."""

    def __init__(self, auth_provider=None):
        """
        Initialize the API client.

        Args:
            auth_provider: Optional provider of API keys
        """
        self.auth_provider = auth_provider

    async def call_openai_api(
            self,
            messages,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call OpenAI Chat API with unified error handling.

        Args:
            messages: List of messages
            model: Optional model override
            temperature: Optional temperature
            max_tokens: Optional token limit

        Returns:
            Response with success status and data
        """
        # Get the API key (user-provided only)
        openai_api_key = settings.auth.OPENAI_API_KEY

        if not openai_api_key:
            logger.error("OpenAI API key not available. User must provide an API key.")
            return {
                "success": False,
                "error": "OpenAI API key is missing. Please provide your API key at /api-keys/openai",
                "message": "I'm sorry, but I can't process your request because the OpenAI API key is missing."
            }

        # Use provided parameters or defaults from settings
        if model is None:
            model = settings.openai.GPT_MODEL
        if temperature is None:
            temperature = settings.openai.GPT_TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.openai.GPT_MAX_TOKENS

        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                settings.openai.OPENAI_API_URL,
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 401:
                logger.error("Authentication failed with OpenAI API. Please check your API key.")
                return {
                    "success": False,
                    "error": "Invalid OpenAI API key. Please provide a valid API key at /api-keys/openai",
                    "message": "I'm sorry, but I can't process your request because the OpenAI API key is invalid."
                }

            response.raise_for_status()
            result = response.json()

            return {
                "success": True,
                "message": result["choices"][0]["message"]["content"],
                "model": model,
                "usage": result.get("usage", {}),
            }

        except requests.exceptions.RequestException as e:
            error_message = f"Error from OpenAI API: {str(e)}"
            if hasattr(e, "response") and e.response is not None and e.response.text:
                error_message += f" Response: {e.response.text}"
            logger.error(error_message)
            return {"success": False, "error": error_message}
        except Exception as e:
            error_message = f"Error generating chat response: {str(e)}"
            logger.error(error_message)
            return {"success": False, "error": error_message}

    async def call_huggingface_api(
            self,
            model_id: str,
            audio_content: bytes,
            content_type: str,
            max_retries: Optional[int] = None,
            backoff_factor: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call Hugging Face Inference API with unified error handling and retry logic.

        Args:
            model_id: HF model identifier
            audio_content: Audio bytes
            content_type: MIME type
            max_retries: Optional retry count
            backoff_factor: Optional backoff multiplier

        Returns:
            Response with transcription or error
        """
        # Get the API token (user-provided only)
        huggingface_token = settings.auth.HUGGINGFACE_TOKEN

        if not huggingface_token:
            logger.error("HuggingFace token not available. User must provide an API key.")
            return {
                "success": False,
                "error": "HuggingFace API key is missing",
                "text": "Error: HuggingFace API key is missing. Please provide your API key at /api-keys/huggingface"
            }

        # Use default settings if not provided
        if max_retries is None:
            max_retries = settings.stt.SPEECH_RECOGNITION_RETRIES
        if backoff_factor is None:
            backoff_factor = settings.stt.SPEECH_RECOGNITION_BACKOFF_FACTOR

        # Construct the full API URL with the selected model
        api_url = f"{settings.stt.HUGGINGFACE_API_URL}/{model_id}"

        headers = {
            "Authorization": f"Bearer {huggingface_token}",
            "Content-Type": content_type,
        }

        # Try multiple times with exponential backoff
        for attempt in range(max_retries):
            try:
                logger.info(f"HF API attempt {attempt + 1}/{max_retries} for model {model_id}")

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
                        wait_time = max(3, (backoff_factor ** attempt) * 2)
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
                    wait_time = max(5, (backoff_factor ** attempt) * 2)
                    logger.warning(f"API returned 503, model likely loading. Retry in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Other error
                    logger.error(f"API error: {response.status_code} - {response.text}")

                    if attempt < max_retries - 1:
                        # Wait longer before retrying
                        wait_time = (backoff_factor ** attempt) * 1.5 + 1
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
                wait_time = (backoff_factor ** attempt) * 2 + 3
                if attempt < max_retries - 1:
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

                if attempt < max_retries - 1:
                    # Try again with increased delay
                    wait_time = (backoff_factor ** attempt) + 2
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

    async def call_elevenlabs_api(
            self,
            text: str,
            voice_id: str,
            model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call ElevenLabs API with unified error handling.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            model_id: Optional model identifier

        Returns:
            Response with audio content or error
        """
        # Get the API key (user-provided only)
        elevenlabs_api_key = settings.auth.ELEVENLABS_API_KEY

        headers = {"xi-api-key": elevenlabs_api_key, "Content-Type": "application/json"}

        # Use provided model ID or default from settings
        if model_id is None:
            model_id = settings.tts.TTS_MODEL_ID

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": settings.tts.TTS_DEFAULT_SETTINGS,
        }

        url = f"{settings.tts.ELEVENLABS_API_URL}/{voice_id}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 401:
                logger.error("Authentication failed with ElevenLabs API. Please check your API key.")
                return {
                    "success": False,
                    "error": "Invalid ElevenLabs API key",
                    "message": "Invalid ElevenLabs API key. Please provide a valid API key.",
                }

            response.raise_for_status()

            return {
                "success": True,
                "audio_content": response.content,
            }

        except requests.exceptions.RequestException as e:
            error_message = f"Error generating speech with ElevenLabs: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                error_message += f" Response: {e.response.text}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message,
            }
        except Exception as e:
            error_message = f"Error generating speech: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message,
            }

    async def get_elevenlabs_voices(self) -> Dict[str, Any]:
        """
        Get available voices from ElevenLabs API.

        Returns:
            Response with voices data or error
        """
        # Get the API key (user-provided only)
        elevenlabs_api_key = settings.auth.ELEVENLABS_API_KEY

        if not elevenlabs_api_key:
            logger.error("ElevenLabs API key not available. User must provide an API key.")
            return {
                "success": False,
                "error": "ElevenLabs API key is missing. Please provide your API key at /api-keys/elevenlabs",
            }

        headers = {"xi-api-key": elevenlabs_api_key}

        try:
            response = requests.get(
                "https://api.elevenlabs.io/v1/voices", headers=headers, timeout=10
            )

            if response.status_code == 401:
                logger.error("Authentication failed with ElevenLabs API. Please check your API key.")
                return {
                    "success": False,
                    "error": "Invalid ElevenLabs API key. Please provide a valid API key.",
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
        except Exception as e:
            error_message = f"Error fetching voices: {str(e)}"
            logger.error(error_message)
            return {"success": False, "error": error_message}
