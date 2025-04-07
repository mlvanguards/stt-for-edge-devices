import logging
import asyncio
from typing import Any, Dict, List, Optional
from src.config.settings import settings
logger = logging.getLogger(__name__)

class ChatService:
    """
    Service for handling chat operations with memory management.
    """
    def __init__(self, memory_service=None, conversation_service=None):
        self.memory_service = memory_service
        self.conversation_service = conversation_service
        self.external_api_client = None  # To be injected (e.g. OpenAIGatewayClient)

    async def get_chat_completion(
            self,
            prompt: str,
            conversation_id: Optional[str] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        if conversation_id and not conversation_history and self.conversation_service:
            conversation_history = await self.conversation_service.extract_conversation_context(conversation_id)
        if conversation_history is None:
            conversation_history = []
        original_history_length = len(conversation_history)
        if self.memory_service and len(conversation_history) > 0:
            optimized_history = self.memory_service.optimize_conversation_history(conversation_history)
            optimized_history_length = len(optimized_history)
            logger.info(f"Optimized history from {original_history_length} to {optimized_history_length} messages")
            conversation_history = optimized_history
        else:
            optimized_history_length = original_history_length
        messages = conversation_history + [{"role": "user", "content": prompt}]
        result = await asyncio.to_thread(
            self.external_api_client.chat_completion,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if result["success"]:
            if conversation_id:
                result["conversation_id"] = conversation_id
            result["original_history_length"] = original_history_length
            result["optimized_history_length"] = optimized_history_length
        return result

    async def summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        formatted_conversation = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted_conversation += f"{role}: {content}\n\n"
        summarization_prompt = (
            "Summarize the following conversation in a concise paragraph. "
            "Focus on key topics, questions, and information exchanged. "
            "Keep your summary under 150 words.\n\n"
            f"{formatted_conversation}"
        )
        summary_messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": summarization_prompt},
        ]
        result = await asyncio.to_thread(
            self.external_api_client.chat_completion,
            messages=summary_messages,
            temperature=0.3,
            max_tokens=200
        )
        if result["success"]:
            logger.info(f"Generated summary of {len(messages)} messages")
            return result["message"].strip()
        else:
            logger.error(f"Error generating conversation summary: {result.get('error')}")
            return ""

    async def process_chat_with_conversation(
            self,
            conversation_id: str,
            user_message: str,
            model: Optional[str] = None,
            temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            if not self.conversation_service:
                return {
                    "success": False,
                    "error": "Conversation service not available",
                    "message": "Service configuration error"
                }
            conversation = await self.conversation_service.get_conversation(conversation_id)
            if not conversation:
                return {
                    "success": False,
                    "error": f"Conversation {conversation_id} not found",
                    "message": "Conversation not found"
                }
            result = await self.get_chat_completion(
                prompt=user_message,
                conversation_id=conversation_id,
                model=model,
                temperature=temperature
            )
            if not result["success"]:
                return result
            await self.conversation_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=user_message
            )
            await self.conversation_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=result["message"]
            )
            message_count = conversation.get("message_count", 0) + 2
            if message_count >= settings.memory.MEMORY_SUMMARIZE_THRESHOLD:
                try:
                    await self.conversation_service.summarize_memory(conversation_id)
                except Exception as e:
                    logger.error(f"Error summarizing memory, but continuing: {str(e)}")
            result["conversation_id"] = conversation_id
            messages = await self.conversation_service.get_conversation_messages(conversation_id)
            if messages:
                result["conversation_history"] = []
                for m in messages:
                    if isinstance(m, dict):
                        if m.get("role") != "system":
                            result["conversation_history"].append({
                                "role": m.get("role"),
                                "content": m.get("content")
                            })
                    else:
                        # It's a MessageModel object
                        if m.role != "system":
                            result["conversation_history"].append({
                                "role": m.role,
                                "content": m.content
                            })
            else:
                result["conversation_history"] = []

            return result
        except Exception as e:
            logger.error(f"Error processing chat with conversation: {str(e)}")
            return {"success": False, "error": str(e), "message": "Error processing your message"}