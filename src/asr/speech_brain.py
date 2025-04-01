import logging
from pathlib import Path
from typing import Union, Optional

from speechbrain.inference.ASR import EncoderDecoderASR

from src.asr.base import ASRModelInterface
from src.utils.audio.audio_handling import AudioProcessor

logger = logging.getLogger(__name__)


class SpeechBrainASR(ASRModelInterface):
    """
    SpeechBrain ASR implementation.
    Uses the centralized AudioProcessor for audio handling.
    """

    def __init__(self, model_path: str, save_dir: str, audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize the SpeechBrain ASR model.

        Args:
            model_path: Path to the model
            save_dir: Directory to save the model
            audio_processor: Optional audio processor
        """
        self.model_path = model_path
        self.save_dir = save_dir
        self.model = None
        self.audio_processor = audio_processor or AudioProcessor()

    def load(self, device: str) -> None:
        """
        Load SpeechBrain ASR model.

        Args:
            device: Device to load model on ('cpu' or 'cuda')
        """
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
        """
        Transcribe an audio file using SpeechBrain.

        Args:
            audio_file: Path to the audio file

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Use SpeechBrain's transcribe_file method
        return self.model.transcribe_file(str(audio_file))

    def cleanup(self) -> None:
        """Release resources."""
        self.model = None
        self.audio_processor.clear_memory()
