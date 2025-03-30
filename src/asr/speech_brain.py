import logging
from pathlib import Path
from typing import Union

from speechbrain.inference.ASR import EncoderDecoderASR

from src.asr.base import ASRModelInterface
from src.utils.audio_handling import AudioUtils

logger = logging.getLogger(__name__)


class SpeechBrainASR(ASRModelInterface):
    """SpeechBrain ASR implementation."""

    def __init__(self, model_path: str, save_dir: str):
        self.model_path = model_path
        self.save_dir = save_dir

    def load(self, device: str) -> None:
        """Load SpeechBrain ASR model."""
        try:
            logger.info(f"Loading SpeechBrain ASR model from {self.model_path}")
            self.model = EncoderDecoderASR.from_hparams(
                source=self.model_path,
                savedir=self.save_dir,
                run_opts={"device": device},
            )
            logger.info("SpeechBrain ASR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain ASR model: {str(e)}")
            raise

    def transcribe(self, audio_file: Union[str, Path]) -> str:
        """Transcribe an audio file using SpeechBrain."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.model.transcribe_file(str(audio_file))

    def cleanup(self) -> None:
        """Release resources."""
        self.model = None
        AudioUtils.clear_memory()
