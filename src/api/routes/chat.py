import json
import uuid
import logging
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status

from src.core.database import MongoDB
from src.core.speech.recognition import process_audio_file, clean_transcription
from src.core.speech.tts import synthesize_speech
from src.core.chat import get_chat_completion
from src.models.conversation import ChatResponse
from src.config.settings import (
    DEFAULT_VOICE_ID,
    DEFAULT_SYSTEM_PROMPT,
    ALLOWED_AUDIO_CONTENT_TYPES
)

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def process_audio(
        file: UploadFile = File(...),
        request_data: str = Form(...),
        force_split: bool = Form(False)
):
    """
    Process an audio file with resilient error handling and retries.
    Maintains conversation context between requests.
    Converts the response to speech using ElevenLabs.

    - **file**: Audio file to transcribe
    - **request_data**: JSON string containing conversation_id and/or system_prompt and/or voice_id
    - **force_split**: Boolean flag (not used in direct processing)
    """
    # Validate the uploaded file type
    if file.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}"
        )

    try:
        # Parse the request data
        try:
            data = json.loads(request_data)
            conversation_id = data.get("conversation_id")
            system_prompt = data.get("system_prompt")
            voice_id = data.get("voice_id", DEFAULT_VOICE_ID)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON in request_data"
            )

        # Handle conversation context
        if conversation_id:
            # Use existing conversation
            conversation = await MongoDB.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation {conversation_id} not found"
                )

            system_prompt = conversation["system_prompt"]
            voice_id = conversation.get("voice_id", DEFAULT_VOICE_ID)

            # Get conversation messages
            messages = await MongoDB.get_conversation_messages(conversation_id)

            # Format messages for GPT
            chat_history = [{
                "role": "system",
                "content": system_prompt
            }]

            # Add conversation history (excluding system messages)
            for message in messages:
                if message["role"] != "system":
                    chat_history.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
        else:
            # Create a new conversation
            conversation_id = str(uuid.uuid4())
            system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT

            # Create a new conversation in the database
            await MongoDB.create_conversation(conversation_id, system_prompt, voice_id)

            # Initialize chat history with system message
            chat_history = [{
                "role": "system",
                "content": system_prompt
            }]

        # Read the audio content into memory
        audio_content = await file.read()

        # Process the audio file directly without pydub/ffmpeg
        # This sends the audio directly to HuggingFace
        transcriptions = process_audio_file(audio_content, file.content_type, force_split)

        # Clean up the transcription
        clean_text = clean_transcription(transcriptions)

        # Sort transcriptions by index for response
        sorted_transcriptions = sorted(transcriptions, key=lambda x: x.get("index", 0))
        full_transcription = " ".join([t.get("text", "") for t in sorted_transcriptions])

        # Get response from GPT
        gpt_result = get_chat_completion(clean_text, chat_history)

        if not gpt_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=gpt_result.get("error", "Failed to get chat completion")
            )

        gpt_message = gpt_result["message"]

        # Update conversation history in database
        await MongoDB.add_message(conversation_id, "user", clean_text)
        await MongoDB.add_message(conversation_id, "assistant", gpt_message)

        # Generate TTS audio from the GPT response
        tts_audio_base64 = None
        try:
            success, _, tts_audio_base64 = synthesize_speech(gpt_message, voice_id)
            if not success:
                logger.warning(f"Failed to generate TTS audio: {_}")
        except Exception as tts_error:
            logger.error(f"Error generating TTS: {str(tts_error)}")
            # Continue without TTS if it fails

        # Get updated conversation history for response
        messages = await MongoDB.get_conversation_messages(conversation_id)
        formatted_messages = []
        for message in messages:
            if message["role"] != "system":  # Skip system messages
                formatted_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

        # Build the response
        result = {
            "conversation_id": conversation_id,
            "transcription": clean_text,
            "raw_transcription": full_transcription,
            "segment_transcriptions": sorted_transcriptions,
            "num_segments": len(sorted_transcriptions),
            "response": gpt_message,
            "model": gpt_result["model"],
            "usage": gpt_result.get("usage", {}),
            "conversation_history": formatted_messages
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
            detail=f"Error processing audio: {str(e)}"
        )