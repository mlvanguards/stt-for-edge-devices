import logging
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.core.interfaces.service import IChatService, ISummarizerService, IMemoryService, \
    IConversationService, IExternalAPIClient

logger = logging.getLogger(__name__)


class ChatService(IChatService, ISummarizerService):
    """
    Service for handling chat operations with memory management.
    Implements both IChatService and ISummarizerService interfaces.
    """

    def __init__(
            self,
            external_api_client: IExternalAPIClient,
            memory_service: Optional[IMemoryService] = None,
            conversation_service: Optional[IConversationService] = None
    ):
        """
        Initialize with dependencies injected.

        Args:
            external_api_client: Client for external API interactions
            memory_service: Optional memory service for optimization
            conversation_service: Optional conversation service for storage
        """
        self.external_api_client = external_api_client
        self.memory_service = memory_service
        self.conversation_service = conversation_service

        # If memory service exists, set this service as its summarizer
        if self.memory_service and hasattr(self.memory_service, "summarizer_service"):
            self.memory_service.summarizer_service = self

    async def get_chat_completion(
            self,
            prompt: str,
            conversation_id: Optional[str] = None,
            conversation_history: Optional[List[Dict[str, str]]] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a response from OpenAI's GPT model with optimized conversation history.

        Args:
            prompt: User's input text
            conversation_id: Optional conversation ID to fetch history from
            conversation_history: Optional explicit conversation history
            model: Optional model override
            temperature: Optional temperature setting
            max_tokens: Optional max tokens setting

        Returns:
            Dict with response and metrics
        """
        # Get history from repository if conversation_id is provided
        if conversation_id and not conversation_history and self.conversation_service:
            conversation_history = await self.conversation_service.extract_conversation_context(conversation_id)

        # Use empty history if none provided
        if conversation_history is None:
            conversation_history = []

        # Store original length for metrics
        original_history_length = len(conversation_history)

        # Optimize conversation history if memory service is available
        if self.memory_service and len(conversation_history) > 0:
            optimized_history = self.memory_service.optimize_conversation_history(conversation_history)
            optimized_history_length = len(optimized_history)
            logger.info(f"Optimized history from {original_history_length} to {optimized_history_length} messages")
            conversation_history = optimized_history
        else:
            optimized_history_length = original_history_length

        # Prepare the messages for the chat API
        messages = conversation_history + [{"role": "user", "content": prompt}]

        # Call OpenAI API through the injected client
        result = await self.external_api_client.call_openai_api(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Add conversation details to the result
        if result["success"]:
            if conversation_id:
                result["conversation_id"] = conversation_id

            # Add memory optimization metrics
            result["original_history_length"] = original_history_length
            result["optimized_history_length"] = optimized_history_length

        return result

    async def summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Create a concise summary of conversation messages using OpenAI.
        Implements ISummarizerService interface.

        Args:
            messages: List of conversation messages to summarize

        Returns:
            Summary text or empty string if failed
        """
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

        # Call OpenAI with the summarization prompt
        summary_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes conversations.",
            },
            {"role": "user", "content": summarization_prompt},
        ]

        result = await self.external_api_client.call_openai_api(
            messages=summary_messages,
            temperature=0.3,  # Lower temperature for more consistent summaries
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
        """
        Process a user message in a conversation context using repositories.

        Args:
            conversation_id: The conversation ID
            user_message: The user's input text
            model: Optional model override
            temperature: Optional temperature setting

        Returns:
            Dict with response and metrics
        """
        try:
            # Validate conversation exists
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

            # Get chat completion
            result = await self.get_chat_completion(
                prompt=user_message,
                conversation_id=conversation_id,
                model=model,
                temperature=temperature
            )

            if not result["success"]:
                return result

            # Add user message and assistant response to conversation
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

            # Check if we need to update the memory summary
            # Use message_count safely with fallback to zero
            message_count = conversation.get("message_count", 0) + 2  # Added user and assistant messages

            if message_count >= settings.memory.MEMORY_SUMMARIZE_THRESHOLD:
                try:
                    # Update memory summary in the background - with error handling
                    await self.conversation_service.summarize_memory(conversation_id)
                except Exception as e:
                    # Log but don't fail the whole request if memory summarization fails
                    logger.error(f"Error summarizing memory, but continuing: {str(e)}")

            # Add additional conversation information to the result
            result["conversation_id"] = conversation_id

            # Get updated message history for the response
            messages = await self.conversation_service.get_conversation_messages(conversation_id)

            # Safe handling for messages
            if messages:
                result["conversation_history"] = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                    if m["role"] != "system"
                ]
            else:
                result["conversation_history"] = []

            return result

        except Exception as e:
            logger.error(f"Error processing chat with conversation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Error processing your message"
            }
