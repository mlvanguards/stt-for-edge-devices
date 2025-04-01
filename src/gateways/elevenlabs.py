from typing import Optional, Dict, Any
import requests
import logging
from requests.exceptions import RequestException, HTTPError

from src.config.settings import settings
from src.errors import ExternalServiceAPIError

logger = logging.getLogger(__name__)


class ElevenLabsGatewayClient:
    """Client for interacting with ElevenLabs API"""

    def __init__(self, base_url: str = "https://api.elevenlabs.io/v1"):
        self._base_url = base_url
        self._api_key = settings.auth.ELEVENLABS_API_KEY

    def _make_request(
            self,
            path: str,
            method: str = 'GET',
            data: Optional[Dict[Any, Any]] = None,
            headers: Optional[Dict[Any, str]] = None,
            params: Optional[Dict[Any, str]] = None,
            return_raw: bool = False
    ):
        default_headers = {"xi-api-key": self._api_key, "Content-Type": "application/json"}

        if headers:
            default_headers.update(headers)

        try:
            response = requests.request(
                url=f'{self._base_url}/{path}',
                method=method,
                json=data,
                headers=default_headers,
                params=params,
                timeout=(10.0, 300.0)  # (connect timeout, read timeout)
            )
            response.raise_for_status()

            if return_raw:
                return response.content

            if response.headers.get('Content-Type', '').startswith('application/json'):
                return response.json()
            return None

        except (RequestException, ConnectionError):
            raise ExternalServiceAPIError(503, "Service Unavailable")
        except HTTPError as e:
            if e.response.status_code == 401:
                raise ExternalServiceAPIError(401, "Invalid ElevenLabs API key")
            raise ExternalServiceAPIError(e.response.status_code, str(e))

    def text_to_speech(self, text: str, voice_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            model_id: Optional model identifier

        Returns:
            Dict with success status and audio content
        """
        if not text:
            return {
                "success": False,
                "error": "No text provided",
                "message": "No text provided for speech synthesis",
            }

        # Use provided model ID or default from settings
        _model_id = model_id or settings.tts.TTS_MODEL_ID

        payload = {
            "text": text,
            "model_id": _model_id,
            "voice_settings": settings.tts.TTS_DEFAULT_SETTINGS,
        }

        path = f"text-to-speech/{voice_id}"

        try:
            audio_content = self._make_request(path=path, method="POST", data=payload, return_raw=True)

            return {
                "success": True,
                "audio_content": audio_content,
            }
        except ExternalServiceAPIError as e:
            logger.error(f"Error generating speech with ElevenLabs: {str(e)}")
            return {
                "success": False,
                "error": f"API error {e.code}: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }

    def get_voices(self) -> Dict[str, Any]:
        """
        Get available voices from ElevenLabs API.

        Returns:
            Dict with voices data or error message
        """
        try:
            voices_data = self._make_request(path="voices")

            # Format the voice data for the API response
            voices = []
            if voices_data and "voices" in voices_data:
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

        except ExternalServiceAPIError as e:
            logger.error(f"Error fetching voices from ElevenLabs: {str(e)}")
            return {"success": False, "error": f"API error {e.code}: {str(e)}"}
        except Exception as e:
            logger.error(f"Error fetching voices: {str(e)}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
