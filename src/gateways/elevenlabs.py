from typing import Optional, Dict, Any
import requests
from requests.exceptions import RequestException, HTTPError

from src.config.settings import settings
from src.erros import ExternalServiceAPIError

class ElevenLabsGatewayClient:

    def __init__(self, base_host: str = settings.tts.ELEVENLABS_API_URL):
        self._base_host = base_host

    def _make_request(
        self,
        path: str,
        method: str = 'GET',
        data: Optional[Dict[Any, str]] = None,
        headers: Optional[Dict[Any, str]] = None,
        params: Optional[Dict[Any, str]] = None
    ):
        default_headers = {"xi-api-key": self._api_key, "Content-Type": "application/json"}
        
        if headers:
            default_headers.update(headers)

        try:
            response = requests.request(
                url=f'{self._base_host}/{path}',
                method=method,
                json=data,
                headers=default_headers,
                params=params,
                timeout=(10.0, 300.0)  # (connect timeout, read timeout)
            )
            response.raise_for_status()
        except (RequestException, ConnectionError):
            raise ExternalServiceAPIError(503, "Service Unavailable")
        except HTTPError as e:
            raise e

        if response.text:
            return response.json()
        return None

    def text_to_speech(self, text: str, voice_id: str, model_id: Optional[str] = None) -> bytes:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            model_id: Optional model identifier

        Returns:
            Audio content as bytes
        """

        # Use provided model ID or default from settings
        if model_id is None:
            model_id = settings.tts.TTS_MODEL_ID

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": settings.tts.TTS_DEFAULT_SETTINGS,
        }

        return self._make_request("text-to-speech", "POST", payload=payload)

    def get_voices(self) -> Dict[str, Any]:
        """
        Get available voices from ElevenLabs API.

        Returns:
            Dictionary containing voices data
        """
        
        try:
            response = requests.get(
                f"{self._base_host}/voices",
                timeout=10
            )
            response.raise_for_status()
            
            voices_data = response.json()
            
            # Format the voice data
            voices = [
                {
                    "voice_id": voice["voice_id"],
                    "name": voice["name"],
                    "preview_url": voice.get("preview_url", None),
                    "category": voice.get("category", "premium"),
                }
                for voice in voices_data.get("voices", [])
            ]
            
            return {"voices": voices}
            
        except HTTPError as e:
            if e.response.status_code == 401:
                raise ExternalServiceAPIError(401, "Invalid ElevenLabs API key")
            raise ExternalServiceAPIError(e.response.status_code, str(e))