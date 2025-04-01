import logging
from typing import Dict, List, Any
from src.config.settings import settings
logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self, summarizer_service=None):
        self.summarizer_service = summarizer_service
        self.memory_max_messages = settings.memory.MEMORY_MAX_MESSAGES
        self.summarize_threshold = settings.memory.MEMORY_SUMMARIZE_THRESHOLD

    def optimize_conversation_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if messages is None:
            logger.warning("Received None messages in optimize_conversation_history")
            return []
        if not isinstance(messages, list):
            logger.warning(f"Expected list of messages but got {type(messages)}")
            return []
        if len(messages) == 0:
            return []
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
        if len(conversation) <= self.memory_max_messages:
            return system_messages + conversation
        for msg in system_messages:
            content = msg.get("content", "")
            if content and "Previous conversation summary:" in content:
                keep_count = min(self.memory_max_messages, len(conversation))
                recent_messages = conversation[-keep_count:]
                return system_messages + recent_messages
        split_point = max(0, len(conversation) - self.memory_max_messages + 2)
        older_messages = conversation[:split_point]
        recent_messages = conversation[split_point:]
        if older_messages and len(older_messages) >= self.summarize_threshold:
            total_msgs = len(older_messages)
            last_msg = older_messages[-1].get('content', '')
            if len(last_msg) > 100:
                last_msg = last_msg[:97] + "..."
            summary_message = {
                "role": "system",
                "content": f"Previous conversation with {total_msgs} messages. Most recent topic: {last_msg}",
            }
            return system_messages + [summary_message] + recent_messages
        return system_messages + recent_messages

    def _get_basic_summary(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return "No previous conversation."
        total_msgs = len(messages)
        user_topics = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                text = msg.get("content", "")
                snippet = text[:40] + "..." if len(text) > 40 else text
                user_topics.append(snippet)
                if len(user_topics) >= 3:
                    break
        if user_topics:
            topics_text = "; ".join(user_topics[-3:])
            return f"Previous conversation with {total_msgs} messages. User topics included: {topics_text}"
        else:
            return f"Previous conversation with {total_msgs} messages."
