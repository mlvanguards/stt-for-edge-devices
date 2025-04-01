import logging
from fastapi import APIRouter, Form, HTTPException, status, Depends
from src.config.settings import settings
from src.services.conversation import ConversationService
from src.dependencies import get_conversation_service

router = APIRouter(tags=["models"])

@router.get("/available_models")
async def get_available_models():
    """Get the list of available speech-to-text models"""
    try:
        return {
            "models": settings.stt.AVAILABLE_STT_MODELS,
            "default_model": settings.stt.DEFAULT_STT_MODEL_ID
        }
    except Exception as e:
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating conversation model: {str(e)}",
        )
