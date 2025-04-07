from typing import Optional, Dict, Any, List
import requests
from requests.exceptions import RequestException, HTTPError

from src.config.settings import settings
from src.errors import ExternalServiceAPIError


class OpenAIGatewayClient:
    """Client for interacting with OpenAI API"""

    def __init__(self, base_url: str = "https://api.openai.com/v1"):
        self._base_url = base_url
        self._api_key = settings.auth.OPENAI_API_KEY

    def _make_request(
            self,
            path: str,
            method: str = 'POST',
            data: Optional[Dict[Any, Any]] = None,
            headers: Optional[Dict[Any, str]] = None,
            params: Optional[Dict[Any, str]] = None
    ):
        default_headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        if headers:
            default_headers.update(headers)

        try:
            response = requests.request(
                url=f'{self._base_url}/{path}',
                method=method,
                json=data,
                headers=default_headers,
                params=params,
                timeout=(10.0, 60.0)  # (connect timeout, read timeout)
            )
            response.raise_for_status()
        except (RequestException, ConnectionError):
            raise ExternalServiceAPIError(503, "Service Unavailable")
        except HTTPError as e:
            if e.response.status_code == 401:
                raise ExternalServiceAPIError(401, "Invalid OpenAI API key")
            raise ExternalServiceAPIError(e.response.status_code, str(e))

        return response.json()

    def chat_completion(
            self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a chat completion from OpenAI API.

        Args:
            messages: List of message objects with role and content
            model: Optional model override
            temperature: Optional temperature setting
            max_tokens: Optional max tokens setting

        Returns:
            OpenAI API response
        """
        # Use provided parameters or defaults from settings
        _model = model or settings.openai.GPT_MODEL
        _temperature = temperature or settings.openai.GPT_TEMPERATURE
        _max_tokens = max_tokens or settings.openai.GPT_MAX_TOKENS

        payload = {
            "model": _model,
            "messages": messages,
            "temperature": _temperature,
            "max_tokens": _max_tokens,
        }

        try:
            result = self._make_request(
                path="chat/completions",
                method="POST",
                data=payload
            )

            return {
                "success": True,
                "message": result["choices"][0]["message"]["content"],
                "model": _model,
                "usage": result.get("usage", {}),
            }
        except ExternalServiceAPIError as e:
            return {
                "success": False,
                "error": f"API error {e.code}: {str(e)}",
                "message": f"Failed to get response from OpenAI: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "message": "An unexpected error occurred"
            }
