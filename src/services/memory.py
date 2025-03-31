import logging
from typing import Dict, List, Any, Optional

from src.config.settings import settings
from src.core.interfaces.service import IMemoryService, ISummarizerService

logger = logging.getLogger(__name__)


class MemoryService(IMemoryService):
    """
    Service for managing conversation memory and summarization.
    Implements the IMemoryService interface.
    """

    def __init__(self, summarizer_service: Optional[ISummarizerService] = None):
        """
        Initialize the memory service with dependencies.

        Args:
            summarizer_service: Optional service for creating summaries
        """
        self.summarizer_service = summarizer_service
        self.memory_max_messages = settings.memory.MEMORY_MAX_MESSAGES
        self.summarize_threshold = settings.memory.MEMORY_SUMMARIZE_THRESHOLD

    def optimize_conversation_history(
            self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize the conversation history for the LLM by:
        1. Keeping system messages
        2. Using memory summaries for older parts of the conversation
        3. Keeping recent messages in full

        Args:
            messages: The full conversation history

        Returns:
            An optimized list of messages for the LLM
        """
        # Defensive checks to prevent NoneType errors
        if messages is None:
            logger.warning("Received None messages in optimize_conversation_history")
            return []

        if not isinstance(messages, list):
            logger.warning(f"Expected list of messages but got {type(messages)}")
            return []

        if len(messages) == 0:
            return []

        # Separate system messages from the conversation
        system_messages = []
        conversation = []

        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning(f"Unexpected message format: {msg}")
                continue

            role = msg.get("role")
            if role == "system":
                system_messages.append(msg)
            elif role in ["user", "assistant"]:
                conversation.append(msg)
            else:
                logger.warning(f"Unknown message role: {role}")

        # If conversation is short enough, return everything
        if len(conversation) <= self.memory_max_messages:
            return system_messages + conversation

        # If memory is already included in system prompt, we can just truncate
        for msg in system_messages:
            content = msg.get("content", "")
            if content and "Previous conversation summary:" in content:
                # Keep more recent messages since we have a memory summary
                keep_count = min(self.memory_max_messages, len(conversation))
                recent_messages = conversation[-keep_count:]
                return system_messages + recent_messages

        # If no memory in system message, we need to truncate more aggressively
        split_point = max(
            0, len(conversation) - self.memory_max_messages + 2
        )  # Keep n-2 recent messages (leave room for summary)
        older_messages = conversation[:split_point]
        recent_messages = conversation[split_point:]

        # If we have older messages to summarize
        if older_messages and len(older_messages) >= self.summarize_threshold:
            # Create a basic summary if we can't get a real one
            total_msgs = len(older_messages)
            last_msg = older_messages[-1].get('content', '')
            if len(last_msg) > 100:
                last_msg = last_msg[:97] + "..."

            summary_message = {
                "role": "system",
                "content": f"Previous conversation with {total_msgs} messages. Most recent topic: {last_msg}",
            }

            return system_messages + [summary_message] + recent_messages

        # If summarization failed or not enough older messages, just return recent ones
        return system_messages + recent_messages

    def _get_basic_summary(self, messages: List[Dict[str, Any]]) -> str:
        """
        Create a basic summary when async summarizer is unavailable.

        Args:
            messages: List of messages to summarize

        Returns:
            A basic summary string
        """
        # Defensive checks
        if not messages:
            return "No previous conversation."

        total_msgs = len(messages)
        user_topics = []

        # Extract a few user message snippets
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            if msg.get("role") == "user":
                text = msg.get("content", "")
                # Take first 40 chars of user messages
                snippet = text[:40] + "..." if len(text) > 40 else text
                user_topics.append(snippet)
                if len(user_topics) >= 3:  # Limit to last 3 user topics
                    break

        # Format them into a basic summary
        if user_topics:
            topics_text = "; ".join(user_topics[-3:])  # Take last 3 topics
            return f"Previous conversation with {total_msgs} messages. User topics included: {topics_text}"
        else:
            return f"Previous conversation with {total_msgs} messages."
