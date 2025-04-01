import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from src.config.settings import settings
logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self, conversation_repo, message_repo, memory_repo, memory_service=None, external_api_client=None):
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
        try:
            conversation_id = str(uuid.uuid4())
            _system_prompt = system_prompt or settings.conversation.DEFAULT_SYSTEM_PROMPT
            _voice_id = voice_id or settings.tts.DEFAULT_VOICE_ID
            _stt_model_id = stt_model_id or settings.stt.DEFAULT_STT_MODEL_ID

            # Create a ConversationModel instance instead of a dictionary
            from src.models.conversation import ConversationModel
            conversation = ConversationModel(
                id=conversation_id,
                conversation_id=conversation_id,
                system_prompt=_system_prompt,
                voice_id=_voice_id,
                stt_model_id=_stt_model_id,
                message_count=0,
                memory_optimized=False
            )

            created_conversation = await self.conversation_repo.create(conversation)

            if created_conversation:
                await self.add_message(
                    conversation_id=conversation_id,
                    role="system",
                    content=_system_prompt
                )
                logger.info(f"Created conversation: {conversation_id}")
                return created_conversation
            else:
                logger.error("Failed to create conversation")
                return None
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            return None

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return await self.conversation_repo.get_by_conversation_id(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> bool:
        try:
            conversation_deleted = await self.conversation_repo.delete(conversation_id)
            if conversation_deleted:
                await self.message_repo.delete_by_conversation_id(conversation_id)
                await self.memory_repo.delete_by_conversation_id(conversation_id)
                logger.info(f"Deleted conversation: {conversation_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            return False

    async def list_conversations(self, limit: int = 10, skip: int = 0) -> Dict[str, Any]:
        conversations = await self.conversation_repo.list(skip=skip, limit=limit)
        total = len(conversations)
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
        try:
            conversation = await self.conversation_repo.get_by_conversation_id(conversation_id)
            if not conversation:
                logger.error(f"Cannot add message to non-existent conversation: {conversation_id}")
                return None

            # Create and insert a message document directly to avoid model conversion issues
            import uuid

            # Generate a new unique ID for this message
            message_id = str(uuid.uuid4())

            # Create document to insert directly into MongoDB
            message_doc = {
                "_id": message_id,  # Use a new unique ID, not the conversation_id
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "timestamp": timestamp or datetime.utcnow().isoformat(),
                "importance": importance
            }

            # Insert directly using the collection
            collection = self.message_repo._collection
            result = await collection.insert_one(message_doc)

            if result.acknowledged:
                await self.conversation_repo.increment_message_count(conversation_id)
                logger.info(f"Added {role} message to conversation: {conversation_id}")
                return message_doc
            else:
                logger.error(f"Failed to add message to conversation: {conversation_id}")
                return None
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return None

    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        return await self.message_repo.get_by_conversation_id(conversation_id)

    async def extract_conversation_context(self, conversation_id: str) -> List[Dict]:
        conversation = await self.conversation_repo.get_by_conversation_id(conversation_id)
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found when extracting context")
            return []

        system_prompt = conversation.get("system_prompt", settings.conversation.DEFAULT_SYSTEM_PROMPT)
        chat_history = [{"role": "system", "content": system_prompt}]

        memory_summary = None
        try:
            if settings.memory.MEMORY_ENABLED:
                memory_doc = await self.memory_repo.get_by_conversation_id(conversation_id)
                if memory_doc and "summary" in memory_doc:
                    memory_summary = memory_doc["summary"]
        except Exception as e:
            logger.error(f"Error retrieving memory summary: {str(e)}")

        if memory_summary:
            chat_history.append({
                "role": "system",
                "content": f"Previous conversation summary: {memory_summary}"
            })

        try:
            messages = await self.message_repo.get_by_conversation_id(conversation_id)
        except Exception as e:
            logger.error(f"Error retrieving messages: {str(e)}")
            messages = []

        if messages:
            for message in messages:
                if isinstance(message, dict):
                    # It's already a dictionary
                    if message.get("role") != "system":
                        chat_history.append({
                            "role": message.get("role", "user"),
                            "content": message.get("content", "")
                        })
                else:
                    # It's a MessageModel object - use attribute access
                    if message.role != "system":
                        chat_history.append({
                            "role": message.role,
                            "content": message.content
                        })

        return chat_history

    async def update_conversation(self, conversation_id: str, updates: Dict[str, Any]) -> bool:
        try:
            conversation = await self.conversation_repo.get_by_conversation_id(conversation_id)
            if not conversation:
                logger.error(f"Cannot update non-existent conversation: {conversation_id}")
                return False
            return await self.conversation_repo.update(conversation_id, updates)
        except Exception as e:
            logger.error(f"Error updating conversation: {str(e)}")
            return False

    async def summarize_memory(self, conversation_id: str) -> bool:
        if not self.external_api_client or not settings.memory.MEMORY_ENABLED:
            return False
        try:
            messages = await self.message_repo.get_by_conversation_id(conversation_id)
            if len(messages) < settings.memory.MEMORY_SUMMARIZE_THRESHOLD:
                return False

            # Filter out system messages and convert objects to dictionaries
            messages_to_summarize = []
            for msg in messages:
                if isinstance(msg, dict):
                    # It's already a dictionary
                    if msg.get("role") != "system":
                        messages_to_summarize.append(msg)
                else:
                    # It's a MessageModel object
                    if msg.role != "system":
                        messages_to_summarize.append({
                            "role": msg.role,
                            "content": msg.content
                        })

            if len(messages_to_summarize) < settings.memory.MEMORY_SUMMARIZE_THRESHOLD:
                return False

            formatted_conversation = ""
            for msg in messages_to_summarize:
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

            import asyncio
            result = await asyncio.to_thread(
                self.external_api_client.chat_completion,
                messages=summary_messages,
                temperature=0.3,
                max_tokens=200
            )

            if result["success"]:
                summary_text = result["message"].strip()
                updated = await self.memory_repo.update_summary(conversation_id, summary_text)
                if updated:
                    await self.conversation_repo.update(conversation_id, {"memory_optimized": True})
                    logger.info(f"Updated memory summary for conversation: {conversation_id}")
                    return True

            return False
        except Exception as e:
            logger.error(f"Error updating memory summary: {str(e)}")
            return False
