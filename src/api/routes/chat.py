import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.config.settings import settings
from src.core.chat import get_chat_completion
from src.core.database import MongoDB
from src.models.responses import ChatResponse
from src.services.recognition import SpeechRecognitionService
from src.services.tts import synthesize_speech

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


async def extract_conversation_context(conversation_id: str) -> List[Dict]:
    """
    Extract conversation context with simple message structure
    """
    # Get conversation details
    conversation = await MongoDB.get_conversation(conversation_id)
    if not conversation:
        return []

    # Get messages
    messages = await MongoDB.get_conversation_messages(conversation_id)

    # Format messages for GPT
    system_prompt = conversation["system_prompt"]

    chat_history = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for message in messages:
        if message["role"] != "system":
            chat_history.append(
                {"role": message["role"], "content": message["content"]}
            )

    return chat_history


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
    recognition_service = SpeechRecognitionService()
    # Validate the uploaded file type
    if file.content_type not in settings.audio.ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}",
        )

    # Validate model_id if provided
    if model_id and model_id not in [
        model["id"] for model in settings.stt.AVAILABLE_STT_MODELS
    ]:
        valid_models = [model["id"] for model in settings.stt.AVAILABLE_STT_MODELS]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model ID: {model_id}. Valid options: {valid_models}",
        )

    try:
        # Use parameters directly
        _conversation_id = conversation_id
        _voice_id = voice_id if voice_id is not None else settings.tts.DEFAULT_VOICE_ID
        _model_id = model_id  # Can be None

        # Handle conversation context
        if _conversation_id:
            # Use existing conversation
            conversation = await MongoDB.get_conversation(_conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation {_conversation_id} not found",
                )

            _system_prompt = conversation["system_prompt"]
            _voice_id = conversation.get("voice_id", settings.tts.DEFAULT_VOICE_ID)

            # If no model_id was specified in request, use the one from conversation if available
            if not _model_id and "stt_model_id" in conversation:
                _model_id = conversation["stt_model_id"]

            # Get conversation context
            chat_history = await extract_conversation_context(_conversation_id)
        else:
            # Create a new conversation
            _conversation_id = str(uuid.uuid4())
            _system_prompt = settings.conversation.DEFAULT_SYSTEM_PROMPT

            # Create a new conversation in the database
            await MongoDB.create_conversation(
                _conversation_id, _system_prompt, _voice_id, _model_id
            )

            # Initialize chat history with system message - no timestamp needed for LLM
            chat_history = [{"role": "system", "content": _system_prompt}]

        # Read the audio content into memory
        audio_content = await file.read()

        # Process the audio file with the selected model - either from request, conversation, or default
        transcriptions = recognition_service.process_audio_file(
            audio_content=audio_content,
            content_type=file.content_type,
            model_id=_model_id,
        )

        # Clean up the transcription
        clean_text = recognition_service.clean_transcription(
            transcriptions=transcriptions
        )

        # Sort transcriptions by index for response
        sorted_transcriptions = sorted(transcriptions, key=lambda x: x.get("index", 0))
        full_transcription = " ".join(
            [t.get("text", "") for t in sorted_transcriptions]
        )

        # Get response from GPT with memory optimization
        gpt_result = get_chat_completion(clean_text, chat_history)

        if not gpt_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=gpt_result.get("error", "Failed to get chat completion"),
            )

        gpt_message = gpt_result["message"]

        # Update conversation history in database - use string timestamps
        current_time = datetime.utcnow().isoformat()
        await MongoDB.add_message(
            _conversation_id, "user", clean_text, timestamp=current_time
        )
        await MongoDB.add_message(
            _conversation_id,
            "assistant",
            gpt_message,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Generate TTS audio from the GPT response
        tts_audio_base64 = None
        try:
            success, _, tts_audio_base64 = synthesize_speech(gpt_message, _voice_id)
            if not success:
                logger.warning(f"Failed to generate TTS audio: {_}")
        except Exception as tts_error:
            logger.error(f"Error generating TTS: {str(tts_error)}")
            # Continue without TTS if it fails

        # Get updated conversation history for response
        messages = await MongoDB.get_conversation_messages(_conversation_id)
        formatted_messages = []
        for message in messages:
            if message["role"] != "system":  # Skip system messages
                formatted_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

        model_used = _model_id if _model_id else settings.stt.DEFAULT_STT_MODEL_ID

        # Build the response with all relevant information
        result = {
            "conversation_id": _conversation_id,
            "transcription": clean_text,
            "raw_transcription": full_transcription,
            "segment_transcriptions": sorted_transcriptions,
            "num_segments": len(sorted_transcriptions),
            "response": gpt_message,
            "model": gpt_result["model"],
            "stt_model_used": model_used,
            "usage": gpt_result.get("usage", {}),
            "conversation_history": formatted_messages,
            "memory_stats": {
                "original_history_size": gpt_result.get("original_history_length", 0),
                "optimized_history_size": gpt_result.get("optimized_history_length", 0),
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
