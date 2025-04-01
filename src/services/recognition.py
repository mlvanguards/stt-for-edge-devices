import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.utils.audio.audio_handling import AudioProcessor

logger = logging.getLogger(__name__)

class SpeechRecognitionService:
    def __init__(self, external_api_client, audio_repository, audio_processor: Optional[AudioProcessor] = None):
        self.external_api_client = external_api_client
        self.audio_repository = audio_repository
        self.audio_processor = audio_processor or AudioProcessor()
        self.default_model_id = settings.stt.DEFAULT_STT_MODEL_ID

    async def process_audio_file(
            self,
            audio_content: bytes,
            content_type: str,
            model_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            store_audio: bool = True
    ) -> List[Dict[str, Any]]:
        try:
            selected_model = model_id if model_id else self.default_model_id
            logger.info(f"Processing audio with model {selected_model}")
            if not self.audio_processor.validate_content_type(content_type):
                logger.error(f"Invalid content type: {content_type}")
                return [{"index": 0, "text": "Error: Invalid audio format."}]
            optimized_audio = self.audio_processor.optimize_for_stt(audio_content, content_type)
            if not optimized_audio:
                logger.error("Failed to optimize audio")
                return [{"index": 0, "text": "Error: Failed to process audio file."}]
            audio_id = None
            if store_audio:
                audio_id = await self.audio_repository.save_audio(
                    audio_content=optimized_audio,
                    content_type="audio/wav",
                    conversation_id=conversation_id,
                    ttl_hours=24
                )
            result = await asyncio.to_thread(
                self.external_api_client.speech_to_text,
                model_id=selected_model,
                audio_content=optimized_audio,
                content_type="audio/wav"
            )
            if not result["success"]:
                logger.error(f"API error: {result.get('error')}")
                return [{
                    "index": 0,
                    "text": result["text"],
                    "audio_id": audio_id
                }]
            return [{
                "index": 0,
                "text": result["text"],
                "audio_id": audio_id
            }]
        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {str(e)}")
            return [{"index": 0, "text": f"Error processing audio file: {str(e)}"}]

    def clean_transcription(self, transcriptions: List[Dict[str, Any]]) -> str:
        sorted_transcriptions = sorted(transcriptions, key=lambda x: x.get("index", 0))
        full_text = " ".join([t.get("text", "") for t in sorted_transcriptions])
        clean_text = re.sub(r"\[Segment \d+ transcription failed\]\s*", "", full_text).strip()
        clean_text = re.sub(r"Failed to transcribe audio.*$", "", clean_text).strip()
        clean_text = re.sub(r"Error processing audio.*$", "", clean_text).strip()
        if not clean_text:
            clean_text = "Unable to transcribe audio clearly. Please try again with a clearer recording."
        return clean_text

    async def get_audio_by_id(self, audio_id: str) -> tuple:
        return await self.audio_repository.get_audio(audio_id)

    async def warm_up_inference_api(self) -> None:
        try:
            logger.info(f"Warming up Hugging Face Inference API for model: {self.default_model_id}")
            audio_content = self.audio_processor.create_silent_audio(duration=0.5)
            # Wrap the synchronous call in asyncio.to_thread so we can await it
            result = await asyncio.to_thread(
                self.external_api_client.speech_to_text,
                model_id=self.default_model_id,
                audio_content=audio_content,
                content_type="audio/wav",
                max_retries=1
            )
            logger.info("Waiting for model to initialize...")
            await asyncio.sleep(2)
            logger.info("Warm-up complete")
        except Exception as e:
            logger.warning(f"Failed to warm up inference API: {str(e)}")
