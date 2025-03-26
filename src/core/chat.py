import logging
import requests
from typing import List, Dict, Any, Optional

from src.config.settings import (
    OPENAI_API_URL,
    GPT_MODEL,
    GPT_TEMPERATURE,
    GPT_MAX_TOKENS
)
from src.core.memory import ConversationMemory
from src.utils.api_keys_service import get_openai_api_key

logger = logging.getLogger(__name__)

memory_manager = ConversationMemory()


def get_chat_completion(prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Get a response from OpenAI's GPT model with optimized conversation history.
    Requires user-provided API key.
    Returns the response and usage statistics.
    """
    # Get the API key (user-provided only)
    openai_api_key = get_openai_api_key()

    if not openai_api_key:
        logger.error("OpenAI API key not available. User must provide an API key.")
        return {
            "success": False,
            "error": "OpenAI API key is missing. Please provide your API key at /api-keys/openai",
            "message": "I'm sorry, but I can't process your request because the OpenAI API key is missing. Please provide your API key at /api-keys/openai"
        }

    if conversation_history is None:
        conversation_history = []

    # Optimize conversation history using the memory manager
    original_history_length = len(conversation_history)
    optimized_history = memory_manager.optimize_conversation_history(conversation_history)
    optimized_history_length = len(optimized_history)

    # Prepare the messages for the chat API with optimized history
    messages = optimized_history + [{"role": "user", "content": prompt}]

    # Log message count for monitoring
    logger.info(f"Optimized conversation history from {original_history_length} to {optimized_history_length} messages")

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
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

        if response.status_code == 401:
            logger.error("Authentication failed with OpenAI API. Please check your API key.")
            return {
                "success": False,
                "error": "Invalid OpenAI API key. Please provide a valid API key at /api-keys/openai",
                "message": "I'm sorry, but I can't process your request because the OpenAI API key is invalid. Please provide a valid API key at /api-keys/openai"
            }

        response.raise_for_status()

        result = response.json()

        return {
            "success": True,
            "message": result["choices"][0]["message"]["content"],
            "model": GPT_MODEL,
            "usage": result.get("usage", {}),
            "memory_optimized": True,
            "original_history_length": original_history_length,
            "optimized_history_length": optimized_history_length
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
