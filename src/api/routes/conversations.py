import logging

from fastapi import APIRouter, HTTPException, status

from src.config.settings import settings
from src.models.requests import ConversationCreate
from src.models.responses import ConversationListResponse, ConversationResponse
from src.services.service_container import services

router = APIRouter(tags=["conversations"])
logger = logging.getLogger(__name__)


@router.post(
    "/create_conversation",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(data: ConversationCreate):
    """
    Create a new conversation with a system prompt and voice selection
    """
    conversation_service = services.get("conversation_service")

    try:
        # Create conversation using service
        conversation = await conversation_service.create_conversation(
            system_prompt=data.system_prompt,
            voice_id=data.voice_id,
            stt_model_id=data.stt_model_id
        )

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create conversation"
            )

        return {
            "conversation_id": conversation["conversation_id"],
            "system_prompt": conversation["system_prompt"],
            "voice_id": conversation["voice_id"],
            "stt_model_id": conversation.get("stt_model_id", settings.stt.DEFAULT_STT_MODEL_ID),
            "messages": [],
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating conversation: {str(e)}",
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation_history(conversation_id: str):
    """
    Get the full history of a conversation
    """
    conversation_service = services.get("conversation_service")

    try:
        # Get the conversation from the service
        conversation = await conversation_service.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )

        # Get the messages for the conversation
        messages = await conversation_service.get_conversation_messages(conversation_id)

        # Format messages for response (exclude system messages)
        formatted_messages = []
        for message in messages:
            if message["role"] != "system":  # Skip system messages in the returned history
                formatted_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

        return {
            "conversation_id": conversation_id,
            "system_prompt": conversation["system_prompt"],
            "voice_id": conversation.get("voice_id", settings.tts.DEFAULT_VOICE_ID),
            "stt_model_id": conversation.get("stt_model_id", settings.stt.DEFAULT_STT_MODEL_ID),
            "messages": formatted_messages,
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversation: {str(e)}",
        )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_200_OK)
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and its history
    """
    conversation_service = services.get("conversation_service")

    try:
        # Check if the conversation exists
        conversation = await conversation_service.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )

        # Delete the conversation and its messages
        deleted = await conversation_service.delete_conversation(conversation_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete conversation {conversation_id}",
            )

        return {"message": f"Conversation {conversation_id} deleted successfully"}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting conversation: {str(e)}",
        )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(limit: int = 10, skip: int = 0):
    """
    List all conversations with pagination
    """
    conversation_service = services.get("conversation_service")

    try:
        # Get the conversations from the service with pagination
        result = await conversation_service.list_conversations(limit, skip)

        # Format the conversations for the response
        formatted_conversations = []
        for conv in result["conversations"]:
            formatted_conversations.append(
                {
                    "conversation_id": conv["conversation_id"],
                    "system_prompt": conv["system_prompt"],
                    "voice_id": conv.get("voice_id", settings.tts.DEFAULT_VOICE_ID),
                    "stt_model_id": conv.get("stt_model_id", settings.stt.DEFAULT_STT_MODEL_ID),
                    "created_at": conv["created_at"],
                    "last_updated": conv["last_updated"],
                }
            )

        return {
            "total": result["total"],
            "conversations": formatted_conversations,
            "page": result["page"],
            "limit": result["limit"],
            "pages": result["pages"],
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing conversations: {str(e)}",
        )
