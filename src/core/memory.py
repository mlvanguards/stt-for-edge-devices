import logging
from typing import Any, Dict, List

import requests

from src.config.settings import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Enhanced short-term memory manager for conversations.
    Uses GPT to summarize conversation history.
    """

    def __init__(self):
        """Initialize the memory manager"""
        self.max_messages = settings.memory.MEMORY_MAX_MESSAGES
        self.summarize_threshold = settings.memory.MEMORY_SUMMARIZE_THRESHOLD

    def optimize_conversation_history(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize the conversation history for the LLM by:
        1. Keeping system messages
        2. Summarizing older messages if conversation is long
        3. Keeping recent messages in full

        Args:
            messages: The full conversation history from MongoDB

        Returns:
            An optimized list of messages for the LLM
        """
        if not messages:
            return []

        # Separate system messages from the conversation
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        conversation = [msg for msg in messages if msg["role"] != "system"]

        # If conversation is short enough, return everything
        if len(conversation) <= self.max_messages:
            return system_messages + conversation

        split_point = max(
            0, len(conversation) - self.max_messages + 2
        )  # Keep n-2 recent messages (leave room for summary)
        older_messages = conversation[:split_point]
        recent_messages = conversation[split_point:]

        # If we have older messages to summarize
        if older_messages and len(older_messages) >= self.summarize_threshold:
            # Get a GPT-generated summary of older messages
            summary = self._summarize_with_gpt(older_messages)

            # Add summary as a system message at the beginning
            if summary:
                summary_message = {
                    "role": "system",
                    "content": f"Previous conversation summary: {summary}",
                }
                return system_messages + [summary_message] + recent_messages

        # If summarization failed or not enough older messages, just return recent ones
        return system_messages + recent_messages

    def _summarize_with_gpt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Create a concise summary of older conversation parts using GPT

        Args:
            messages: List of messages to summarize

        Returns:
            A string summary of the conversation
        """
        # Get OpenAI API key from settings
        openai_api_key = settings.openai.OPENAI_API_KEY

        if not openai_api_key:
            logger.error("Cannot summarize: OpenAI API key not configured")
            return ""

        try:
            # Format conversation for summarization
            formatted_conversation = ""
            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                formatted_conversation += f"{role}: {content}\n\n"

            # Create summarization prompt
            summarization_prompt = (
                "Summarize the following conversation in a concise paragraph. "
                "Focus on key topics, questions, and information exchanged. "
                "Keep your summary under 150 words.\n\n"
                f"{formatted_conversation}"
            )

            # Call the OpenAI API
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": settings.openai.GPT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes conversations.",
                    },
                    {"role": "user", "content": summarization_prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            }

            response = requests.post(
                settings.openai.OPENAI_API_URL, headers=headers, json=data
            )
            response.raise_for_status()

            result = response.json()
            summary = result["choices"][0]["message"]["content"].strip()

            logger.info(f"Generated summary of {len(messages)} messages")
            return summary

        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return ""
