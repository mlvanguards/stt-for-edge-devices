import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
import traceback

from src.config.settings import settings
from src.models.responses import ChatResponse
from src.services.service_container import services

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def process_audio(
        file: UploadFile = File(...),
        conversation_id: Optional[str] = Form(None),
        voice_id: Optional[str] = Form(None),
        model_id: Optional[str] = Form(None),
        force_split: bool = Form(False),
):
    """
    Maintains conversation context between requests with enhanced memory.
    Converts the response to speech using ElevenLabs.

    - **file**: Audio file to transcribe
    - **conversation_id**: Optional conversation ID to continue an existing conversation
    - **voice_id**: Optional voice ID for text-to-speech (defaults to system default)
    - **model_id**: Optional model ID for speech recognition (defaults to system default)
    - **force_split**: Boolean flag (not used in direct processing)
    """
    # Get services from the container
    speech_recognition_service = services.get("speech_recognition_service")
    chat_service = services.get("chat_service")
    conversation_service = services.get("conversation_service")
    tts_service = services.get("tts_service")

    # Validate the uploaded file type
    if file.content_type not in settings.audio.ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}",
        )

    # Validate model_id if provided - careful with None values
    if model_id:
        valid_models = [model["id"] for model in settings.stt.AVAILABLE_STT_MODELS]
        if model_id not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model ID: {model_id}. Valid options: {valid_models}",
            )

    try:
        # Use parameters directly
        _conversation_id = conversation_id
        _voice_id = voice_id if voice_id is not None else settings.tts.DEFAULT_VOICE_ID
        _model_id = model_id  # Can be None - will use default in service

        logger.info(
            f"Processing request: conversation_id={_conversation_id}, voice_id={_voice_id}, model_id={_model_id}")

        # Handle conversation context
        if _conversation_id:
            # Use existing conversation
            conversation = await conversation_service.get_conversation(_conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation {_conversation_id} not found",
                )

            _system_prompt = conversation.get("system_prompt", settings.conversation.DEFAULT_SYSTEM_PROMPT)
            _voice_id = conversation.get("voice_id", settings.tts.DEFAULT_VOICE_ID)

            # If no model_id was specified in request, use the one from conversation if available
            if not _model_id and "stt_model_id" in conversation:
                _model_id = conversation.get("stt_model_id")
        else:
            # Create a new conversation
            _system_prompt = settings.conversation.DEFAULT_SYSTEM_PROMPT

            # Create a new conversation in the database
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

        # Read the audio content into memory
        audio_content = await file.read()

        # Process the audio file with the selected model - either from request, conversation, or default
        # Ensure we always have a valid model_id by the time we reach the service
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

        # Safety check for transcriptions
        if not transcriptions:
            logger.warning("Received empty transcriptions from speech recognition service")
            transcriptions = [{"index": 0, "text": "I couldn't understand the audio. Could you please try again?"}]

        # Clean up the transcription
        clean_text = speech_recognition_service.clean_transcription(
            transcriptions=transcriptions
        )

        # Sort transcriptions by index for response
        sorted_transcriptions = sorted(transcriptions, key=lambda x: x.get("index", 0))
        full_transcription = " ".join(
            [t.get("text", "") for t in sorted_transcriptions]
        )

        # Process the chat using the conversation service
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

        # Fetch updated conversation for memory stats
        conversation = await conversation_service.get_conversation(_conversation_id)
        memory_optimized = conversation.get("memory_optimized", False) if conversation else False

        # Generate TTS audio from the GPT response
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
            # Continue without TTS if it fails

        # Build the response with all relevant information
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

        # Add TTS data if available
        if tts_audio_base64:
            result["tts_audio_base64"] = tts_audio_base64

        return result

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())  # Log the full traceback for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio: {str(e)}",
        )


@router.get("/available_stt_models")
async def available_stt_models():
    """
    Get a list of available speech-to-text models
    """
    try:
        return {
            "models": settings.stt.AVAILABLE_STT_MODELS,
            "default_model": settings.stt.DEFAULT_STT_MODEL_ID,
        }
    except Exception as e:
        logger.error(f"Error fetching available models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching models: {str(e)}",
        )
