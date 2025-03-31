import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from src.config.settings import settings
from src.core.interfaces.service import IConversationService, IMemoryService
from src.core.interfaces.repository import IConversationRepository, IMessageRepository, IMemoryRepository
from src.core.interfaces.service import IExternalAPIClient

logger = logging.getLogger(__name__)


class ConversationService(IConversationService):
    """
    Service for managing conversations.
    Implements the IConversationService interface.
    """

    def __init__(
            self,
            conversation_repo: IConversationRepository,
            message_repo: IMessageRepository,
            memory_repo: IMemoryRepository,
            memory_service: Optional[IMemoryService] = None,
            external_api_client: Optional[IExternalAPIClient] = None
    ):
        """
        Initialize the service with repositories and dependencies.

        Args:
            conversation_repo: Repository for conversations
            message_repo: Repository for messages
            memory_repo: Repository for memory summaries
            memory_service: Optional memory service
            external_api_client: Optional external API client
        """
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.memory_repo = memory_repo
        self.memory_service = memory_service
        self.external_api_client = external_api_client

    async def create_conversation(
            self,
            system_prompt: Optional[str] = None,
            voice_id: Optional[str] = None,
            stt_model_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new conversation with default or provided settings.

        Args:
            system_prompt: Optional prompt that defines the AI's behavior
            voice_id: Optional voice ID for text-to-speech
            stt_model_id: Optional model ID for speech recognition

        Returns:
            Newly created conversation or None if creation failed
        """
        try:
            # Generate unique conversation ID
            conversation_id = str(uuid.uuid4())

            # Use provided values or defaults
            _system_prompt = system_prompt or settings.conversation.DEFAULT_SYSTEM_PROMPT
            _voice_id = voice_id or settings.tts.DEFAULT_VOICE_ID
            _stt_model_id = stt_model_id or settings.stt.DEFAULT_STT_MODEL_ID

            # Create conversation entity
            conversation = {
                "conversation_id": conversation_id,
                "system_prompt": _system_prompt,
                "voice_id": _voice_id,
                "stt_model_id": _stt_model_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat(),
                "message_count": 0,
                "memory_optimized": False
            }

            # Save to repository
            created_conversation = await self.conversation_repo.create(conversation)

            if created_conversation:
                # Add system message
                await self.add_message(
                    conversation_id=conversation_id,
                    role="system",
                    content=_system_prompt
                )

                logger.info(f"Created conversation: {conversation_id}")
                return created_conversation
            else:
                logger.error(f"Failed to create conversation")
                return None

        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            return None

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: The conversation identifier

        Returns:
            The conversation if found, None otherwise
        """
        return await self.conversation_repo.get_by_conversation_id(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its associated data.

        Args:
            conversation_id: The conversation identifier

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Delete conversation
            conversation_deleted = await self.conversation_repo.delete(conversation_id)

            if conversation_deleted:
                # Delete messages
                await self.message_repo.delete_by_conversation_id(conversation_id)

                # Delete memory summary
                await self.memory_repo.delete_by_conversation_id(conversation_id)

                logger.info(f"Deleted conversation: {conversation_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            return False

    async def list_conversations(
            self,
            limit: int = 10,
            skip: int = 0
    ) -> Dict[str, Any]:
        """
        List conversations with pagination.

        Args:
            limit: Maximum number of conversations to return
            skip: Number of conversations to skip

        Returns:
            Dictionary with conversations and pagination metadata
        """
        conversations = await self.conversation_repo.list(skip=skip, limit=limit)
        total = len(conversations)  # This is a simplified approach - in a real app, would use a count query

        # Calculate pagination info
        page = (skip // limit) + 1 if limit > 0 else 1
        pages = (total // limit) + (1 if total % limit > 0 else 0)

        return {
            "conversations": conversations,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": pages
        }

    async def add_message(
            self,
            conversation_id: str,
            role: str,
            content: str,
            timestamp: Optional[str] = None,
            importance: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add a message to a conversation.

        Args:
            conversation_id: The conversation identifier
            role: Message role (user, assistant, system)
            content: Message content
            timestamp: Optional timestamp (default: current time)
            importance: Optional importance score for memory prioritization

        Returns:
            The created message if successful, None otherwise
        """
        try:
            # Ensure conversation exists
            conversation = await self.conversation_repo.get_by_conversation_id(conversation_id)
            if not conversation:
                logger.error(f"Cannot add message to non-existent conversation: {conversation_id}")
                return None

            # Create message entity
            message = {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "timestamp": timestamp or datetime.utcnow().isoformat()
            }

            # Add importance if provided
            if importance is not None:
                message["importance"] = importance

            # Save message
            created_message = await self.message_repo.create(message)

            if created_message:
                # Update conversation's last_updated timestamp and increment message count
                await self.conversation_repo.increment_message_count(conversation_id)

                logger.info(f"Added {role} message to conversation: {conversation_id}")
                return created_message
            else:
                logger.error(f"Failed to add message to conversation: {conversation_id}")
                return None

        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return None

    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation.

        Args:
            conversation_id: The conversation identifier

        Returns:
            List of messages
        """
        return await self.message_repo.get_by_conversation_id(conversation_id)

    async def extract_conversation_context(self, conversation_id: str) -> List[Dict]:
        """
        Extract conversation context with simple message structure for the LLM.
        Now improved to include memory summaries and add safety checks.

        Args:
            conversation_id: The conversation identifier

        Returns:
            List of messages formatted for the LLM
        """
        # Get conversation details
        conversation = await self.conversation_repo.get_by_conversation_id(conversation_id)
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found when extracting context")
            return []

        # Format system prompt
        system_prompt = conversation.get("system_prompt", settings.conversation.DEFAULT_SYSTEM_PROMPT)
        chat_history = [{"role": "system", "content": system_prompt}]

        # Check if we have a memory summary for this conversation
        memory_summary = None
        try:
            if settings.memory.MEMORY_ENABLED:
                memory_doc = await self.memory_repo.get_by_conversation_id(conversation_id)
                if memory_doc and "summary" in memory_doc:
                    memory_summary = memory_doc["summary"]
        except Exception as e:
            logger.error(f"Error retrieving memory summary: {str(e)}")
            # Continue without memory if it fails

        # Include memory in system prompt if available
        if memory_summary:
            # Add as a separate system message rather than modifying the original
            chat_history.append({
                "role": "system",
                "content": f"Previous conversation summary: {memory_summary}"
            })

        # Get messages safely
        try:
            messages = await self.message_repo.get_by_conversation_id(conversation_id)
        except Exception as e:
            logger.error(f"Error retrieving messages: {str(e)}")
            messages = []

        # Add conversation history (only if we have messages)
        if messages:
            for message in messages:
                # Skip system messages as we've already added our own
                if message.get("role") != "system":
                    chat_history.append({
                        "role": message.get("role", "user"),  # Default to user if missing
                        "content": message.get("content", "")  # Default to empty if missing
                    })

        return chat_history

    async def update_conversation(
            self,
            conversation_id: str,
            updates: Dict[str, Any]
    ) -> bool:
        """
        Update a conversation's properties.

        Args:
            conversation_id: The conversation identifier
            updates: Dictionary with updated properties

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Ensure conversation exists
            conversation = await self.conversation_repo.get_by_conversation_id(conversation_id)
            if not conversation:
                logger.error(f"Cannot update non-existent conversation: {conversation_id}")
                return False

            # Update the conversation
            return await self.conversation_repo.update(conversation_id, updates)

        except Exception as e:
            logger.error(f"Error updating conversation: {str(e)}")
            return False

    async def summarize_memory(self, conversation_id: str) -> bool:
        """
        Create or update memory summary for a conversation.
        This is a key method for implementing conversation memory.

        Args:
            conversation_id: The conversation identifier

        Returns:
            True if summary created/updated, False otherwise
        """
        if not self.external_api_client or not settings.memory.MEMORY_ENABLED:
            return False

        try:
            # Get messages for the conversation
            messages = await self.message_repo.get_by_conversation_id(conversation_id)

            # If not enough messages to summarize, skip
            if len(messages) < settings.memory.MEMORY_SUMMARIZE_THRESHOLD:
                return False

            # Filter out system messages for summarization
            messages_to_summarize = [msg for msg in messages if msg["role"] != "system"]

            # If still not enough messages after filtering, skip
            if len(messages_to_summarize) < settings.memory.MEMORY_SUMMARIZE_THRESHOLD:
                return False

            # Format conversation for summarization
            formatted_conversation = ""
            for msg in messages_to_summarize:
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
                # Update memory summary in the repository
                summary_text = result["message"].strip()
                updated = await self.memory_repo.update_summary(conversation_id, summary_text)

                if updated:
                    # Mark conversation as memory-optimized
                    await self.conversation_repo.update(conversation_id, {"memory_optimized": True})
                    logger.info(f"Updated memory summary for conversation: {conversation_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error updating memory summary: {str(e)}")
            return False
