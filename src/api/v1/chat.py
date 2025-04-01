import logging
import traceback
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Depends
from src.config.settings import settings
from src.schemas import ChatResponse
from src.dependencies import (
    get_speech_recognition_service,
    get_chat_service,
    get_conversation_service,
    get_tts_service
)
from src.services.recognition import SpeechRecognitionService
from src.services.chat import ChatService
from src.services.conversation import ConversationService
from src.services.tts import TextToSpeechService

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def process_audio(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    voice_id: Optional[str] = Form(None),
    model_id: Optional[str] = Form(None),
    force_split: bool = Form(False),
    speech_recognition_service: SpeechRecognitionService = Depends(get_speech_recognition_service),
    chat_service: ChatService = Depends(get_chat_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    tts_service: TextToSpeechService = Depends(get_tts_service)
):
    if file.content_type not in settings.audio.ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}",
        )
    if model_id:
        valid_models = [model["id"] for model in settings.stt.AVAILABLE_STT_MODELS]
        if model_id not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model ID: {model_id}. Valid options: {valid_models}",
            )
    try:
        _conversation_id = conversation_id
        _voice_id = voice_id if voice_id is not None else settings.tts.DEFAULT_VOICE_ID
        _model_id = model_id  # May be None; will use default later
        logger.info(f"Processing request: conversation_id={_conversation_id}, voice_id={_voice_id}, model_id={_model_id}")
        if _conversation_id:
            try:
                conversation = await conversation_service.get_conversation(_conversation_id)
                if not conversation:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Conversation {_conversation_id} not found",
                    )
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: Please try again later",
                )
            _system_prompt = conversation.get("system_prompt", settings.conversation.DEFAULT_SYSTEM_PROMPT)
            _voice_id = conversation.get("voice_id", settings.tts.DEFAULT_VOICE_ID)
            if not _model_id and "stt_model_id" in conversation:
                _model_id = conversation.get("stt_model_id")
        else:
            _system_prompt = settings.conversation.DEFAULT_SYSTEM_PROMPT
            conversation = await conversation_service.create_conversation(
                system_prompt=_system_prompt,
                voice_id=_voice_id,
                stt_model_id=_model_id
            )
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create conversation",
                )
            _conversation_id = conversation["conversation_id"]
            logger.info(f"Created new conversation with ID: {_conversation_id}")
        audio_content = await file.read()
        if not _model_id:
            _model_id = settings.stt.DEFAULT_STT_MODEL_ID
            logger.info(f"Using default STT model: {_model_id}")
        transcriptions = await speech_recognition_service.process_audio_file(
            audio_content=audio_content,
            content_type=file.content_type,
            model_id=_model_id,
            conversation_id=_conversation_id,
            store_audio=True
        )
        if not transcriptions:
            logger.warning("Received empty transcriptions from speech recognition service")
            transcriptions = [{"index": 0, "text": "I couldn't understand the audio. Could you please try again?"}]
        clean_text = speech_recognition_service.clean_transcription(transcriptions=transcriptions)
        sorted_transcriptions = sorted(transcriptions, key=lambda x: x.get("index", 0))
        full_transcription = " ".join([t.get("text", "") for t in sorted_transcriptions])
        chat_result = await chat_service.process_chat_with_conversation(
            conversation_id=_conversation_id,
            user_message=clean_text
        )
        if not chat_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=chat_result.get("error", "Failed to get chat completion"),
            )
        gpt_message = chat_result["message"]
        conversation = await conversation_service.get_conversation(_conversation_id)
        memory_optimized = conversation.get("memory_optimized", False) if conversation else False
        tts_audio_base64 = None
        try:
            tts_result = await tts_service.synthesize_speech(
                text=gpt_message,
                voice_id=_voice_id,
                conversation_id=_conversation_id
            )
            if tts_result["success"] and "audio_base64" in tts_result:
                tts_audio_base64 = tts_result["audio_base64"]
            else:
                logger.warning(f"Failed to generate TTS audio: {tts_result.get('error')}")
        except Exception as tts_error:
            logger.error(f"Error generating TTS: {str(tts_error)}")
        result = {
            "conversation_id": _conversation_id,
            "transcription": clean_text,
            "raw_transcription": full_transcription,
            "segment_transcriptions": sorted_transcriptions,
            "num_segments": len(sorted_transcriptions),
            "response": gpt_message,
            "model": chat_result.get("model", "unknown"),
            "stt_model_used": _model_id,
            "usage": chat_result.get("usage", {}),
            "conversation_history": chat_result.get("conversation_history", []),
            "memory_stats": {
                "original_history_size": chat_result.get("original_history_length", 0),
                "optimized_history_size": chat_result.get("optimized_history_length", 0),
                "memory_enabled": settings.memory.MEMORY_ENABLED,
                "memory_optimized": memory_optimized
            },
        }
        if tts_audio_base64:
            result["tts_audio_base64"] = tts_audio_base64
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio: {str(e)}",
        )
