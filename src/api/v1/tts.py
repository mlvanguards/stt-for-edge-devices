import logging
from typing import Any, Dict
from fastapi import APIRouter, Body, HTTPException, status, Depends
from src.config.settings import settings
from src.dependencies import get_tts_service
from src.services.tts import TextToSpeechService

router = APIRouter(tags=["text-to-speech"])
logger = logging.getLogger(__name__)

@router.post("/tts_only")
async def text_to_speech_only(
    data: Dict[str, Any] = Body(...),
    tts_service: TextToSpeechService = Depends(get_tts_service)
):
    text = data.get("text")
    voice_id = data.get("voice_id", settings.tts.DEFAULT_VOICE_ID)
    conversation_id = data.get("conversation_id")
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Text is required"
        )
    try:
        result = await tts_service.synthesize_speech(
            text=text,
            voice_id=voice_id,
            conversation_id=conversation_id
        )
        if not result["success"] or "audio_base64" not in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to generate speech"),
            )
        return {"audio_base64": result["audio_base64"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating speech: {str(e)}",
        )

@router.get("/available_voices")
async def available_voices(
    tts_service: TextToSpeechService = Depends(get_tts_service)
):
    try:
        result = await tts_service.get_available_voices()
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Error fetching voices"),
            )
        return {"voices": result["voices"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching voices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching voices: {str(e)}",
        )

@router.get("/audio/{audio_id}")
async def get_audio(
    audio_id: str,
    tts_service: TextToSpeechService = Depends(get_tts_service)
):
    try:
        audio_content, content_type = await tts_service.get_audio_by_id(audio_id)
        if not audio_content or not content_type:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audio file {audio_id} not found",
            )
        from fastapi.responses import Response
        return Response(content=audio_content, media_type=content_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving audio: {str(e)}",
        )
