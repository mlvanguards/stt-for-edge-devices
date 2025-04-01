import logging
from fastapi import APIRouter, HTTPException, status, Depends
from src.config.settings import settings
from src.schemas import ConversationCreate, ConversationListResponse, ConversationResponse
from src.dependencies import get_conversation_service
from src.services.conversation import ConversationService

router = APIRouter(tags=["conversations"])
logger = logging.getLogger(__name__)

@router.post("/create_conversation", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    data: ConversationCreate,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    try:
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
async def get_conversation_history(
        conversation_id: str,
        conversation_service: ConversationService = Depends(get_conversation_service)
):
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )
        messages = await conversation_service.get_conversation_messages(conversation_id)
        formatted_messages = []

        for message in messages:
            # Handle both dictionary and object formats
            if isinstance(message, dict):
                # Dictionary format
                if message.get("role") != "system":
                    formatted_messages.append({
                        "role": message.get("role", "user"),
                        "content": message.get("content", "")
                    })
            else:
                # MessageModel object format
                if message.role != "system":
                    formatted_messages.append({
                        "role": message.role,
                        "content": message.content
                    })

        return {
            "conversation_id": conversation_id,
            "system_prompt": conversation["system_prompt"],
            "voice_id": conversation.get("voice_id", settings.tts.DEFAULT_VOICE_ID),
            "stt_model_id": conversation.get("stt_model_id", settings.stt.DEFAULT_STT_MODEL_ID),
            "messages": formatted_messages,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversation: {str(e)}",
        )

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_200_OK)
async def delete_conversation(
    conversation_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )
        deleted = await conversation_service.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete conversation {conversation_id}",
            )
        return {"message": f"Conversation {conversation_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting conversation: {str(e)}",
        )

@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = 10,
    skip: int = 0,
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    try:
        result = await conversation_service.list_conversations(limit, skip)
        formatted_conversations = []
        for conv in result["conversations"]:
            formatted_conversations.append({
                "conversation_id": conv["conversation_id"],
                "system_prompt": conv["system_prompt"],
                "voice_id": conv.get("voice_id", settings.tts.DEFAULT_VOICE_ID),
                "stt_model_id": conv.get("stt_model_id", settings.stt.DEFAULT_STT_MODEL_ID),
                "created_at": conv["created_at"],
                "last_updated": conv["last_updated"],
            })
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
