import logging
from fastapi import APIRouter, HTTPException, status, Depends, Form
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

# Added model-related endpoints
@router.get("/available_models")
async def get_available_models():
    """Get the list of available speech-to-text models"""
    try:
        return {
            "models": settings.stt.AVAILABLE_STT_MODELS,
            "default_model": settings.stt.DEFAULT_STT_MODEL_ID
        }
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting available models: {str(e)}",
        )

@router.post("/update_model")
async def update_conversation_model(
        conversation_id: str = Form(...),
        model_id: str = Form(...),
        conversation_service: ConversationService = Depends(get_conversation_service)
):
    """Update the STT model for a specific conversation"""
    try:
        # Verify model_id is valid
        valid_models = [model["id"] for model in settings.stt.AVAILABLE_STT_MODELS]
        if model_id not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model ID: {model_id}. Valid options: {valid_models}",
            )

        # Get the conversation
        conversation = await conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found",
            )

        # Update the conversation with the new model_id
        updated = await conversation_service.update_conversation(
            conversation_id=conversation_id,
            updates={"stt_model_id": model_id}
        )

        if not updated:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update model for conversation {conversation_id}",
            )

        return {
            "success": True,
            "conversation_id": conversation_id,
            "model_id": model_id,
            "message": f"Successfully updated model for conversation {conversation_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating conversation model: {str(e)}",
        )
