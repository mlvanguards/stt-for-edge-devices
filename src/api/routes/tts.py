import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, status

from src.config.settings import settings
from src.models.responses import TTSResponse
from src.services.tts import TextToSpeechService

router = APIRouter(tags=["text-to-speech"])
logger = logging.getLogger(__name__)


@router.post("/tts_only", response_model=TTSResponse)
async def text_to_speech_only(data: Dict[str, Any] = Body(...)):
    """
    Convert text to speech using ElevenLabs API without requiring a conversation context

    - **text**: Text to convert to speech
    - **voice_id**: Optional voice ID to use (defaults to system default)
    """
    tts_service = TextToSpeechService()
    text = data.get("text")
    voice_id = data.get("voice_id", settings.tts.DEFAULT_VOICE_ID)

    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Text is required"
        )

    try:
        success, result, audio_base64 = tts_service.synthesize_speech(text, voice_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result,  # This will be the error message
            )

        return {"audio_base64": audio_base64}
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating speech: {str(e)}",
        )


@router.get("/available_voices")
async def available_voices():
    """
    Get a list of available voices from ElevenLabs
    """
    tts_service = TextToSpeechService()
    try:
        result = await tts_service.get_available_voices()

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Error fetching voices"),
            )

        return {"voices": result["voices"]}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error fetching voices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching voices: {str(e)}",
        )
