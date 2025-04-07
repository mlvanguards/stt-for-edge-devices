import logging
import base64
import asyncio
from typing import Dict, Optional, Any, Tuple
from src.config.settings import settings
from src.utils.audio.audio_handling import AudioProcessorMainApp
logger = logging.getLogger(__name__)

class TextToSpeechService:
    def __init__(self, external_api_client, audio_repository, audio_processor: Optional[AudioProcessorMainApp] = None):
        self.external_api_client = external_api_client
        self.audio_repository = audio_repository
        self.audio_processor = audio_processor or AudioProcessorMainApp()
        self.default_voice_id = settings.tts.DEFAULT_VOICE_ID
        self.tts_model_id = settings.tts.TTS_MODEL_ID
        self.voice_cache = None
        self.voice_cache_timestamp = None

    async def synthesize_speech(
            self, text: str, voice_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            return_base64: bool = True
    ) -> Dict[str, Any]:
        if voice_id is None:
            voice_id = self.default_voice_id

        # Get speech from ElevenLabs
        result = await asyncio.to_thread(
            self.external_api_client.text_to_speech,
            text=text,
            voice_id=voice_id,
            model_id=self.tts_model_id
        )

        if not result["success"]:
            logger.error(f"Failed to generate speech: {result.get('error')}")
            return {
                "success": False,
                "error": result.get("error", "Unknown error in speech synthesis"),
                "audio_base64": None,
                "file_id": None
            }

        try:
            audio_content = result["audio_content"]
            # Save the audio directly as MP3
            file_id = await self.audio_repository.save_audio(
                audio_content=audio_content,
                content_type="audio/mpeg",  # Always MP3 from ElevenLabs
                conversation_id=conversation_id,
                ttl_hours=24
            )

            # Prepare response
            response = {
                "success": True,
                "file_id": file_id,
                "error": None
            }

            if return_base64:
                audio_base64 = base64.b64encode(audio_content).decode("utf-8")
                response["audio_base64"] = audio_base64

            return response
        except Exception as e:
            logger.error(f"Error handling speech synthesis: {str(e)}")
            # Try to return base64 if available
            if return_base64 and "audio_content" in result:
                audio_base64 = base64.b64encode(result["audio_content"]).decode("utf-8")
                return {
                    "success": True,
                    "audio_base64": audio_base64,
                    "file_id": None,
                    "error": f"Storage error: {str(e)}"
                }
            return {
                "success": False,
                "error": f"Error processing speech: {str(e)}",
                "audio_base64": None,
                "file_id": None
            }

    async def get_audio_by_id(self, file_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        return await self.audio_repository.get_audio(file_id)

    async def get_available_voices(self) -> Dict[str, Any]:
        if self.voice_cache is not None:
            return self.voice_cache
        result = await asyncio.to_thread(self.external_api_client.get_voices)
        if result.get("success", False):
            self.voice_cache = result
        return result
