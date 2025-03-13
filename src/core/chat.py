import logging
import requests
from typing import List, Dict, Any, Optional

from src.config.settings import (
    OPENAI_API_KEY,
    OPENAI_API_URL,
    GPT_MODEL,
    GPT_TEMPERATURE,
    GPT_MAX_TOKENS
)

logger = logging.getLogger(__name__)


def get_chat_completion(prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Get a response from OpenAI's GPT model
    Returns the response and usage statistics
    """
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not configured")
        return {"success": False, "error": "OpenAI API key not configured"}

    if conversation_history is None:
        conversation_history = []

    # Prepare the messages for the chat API
    messages = conversation_history + [{"role": "user", "content": prompt}]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GPT_MODEL,
        "messages": messages,
        "temperature": GPT_TEMPERATURE,
        "max_tokens": GPT_MAX_TOKENS
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()

        return {
            "success": True,
            "message": result["choices"][0]["message"]["content"],
            "model": GPT_MODEL,
            "usage": result.get("usage", {})
        }
    except requests.exceptions.RequestException as e:
        error_message = f"Error from OpenAI API: {str(e)}"
        if hasattr(e, 'response') and e.response is not None and e.response.text:
            error_message += f" Response: {e.response.text}"
        logger.error(error_message)
        return {"success": False, "error": error_message}
    except Exception as e:
        error_message = f"Error generating chat response: {str(e)}"
        logger.error(error_message)
        return {"success": False, "error": error_message}