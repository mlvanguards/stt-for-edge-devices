from pathlib import Path
from typing import Dict, Optional

from src.asr.base import ASRModelInterface
from src.asr.speech_brain import SpeechBrainASR
from src.utils.audio.audio_process import AudioProcessor


class ASRProcessor:
    """
    Processor for automatic speech recognition using local models.
    Uses the centralized AudioProcessor for all audio processing operations.
    """

    def __init__(self, model: Optional[ASRModelInterface] = None, audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize the ASR processor with model and audio processor.

        Args:
            model: Optional ASR model implementation
            audio_processor: Optional audio processor
        """
        self.model = model
        self.audio_processor = audio_processor or AudioProcessor()
        self._loaded = False

    def load_model(self, device: str = "cpu") -> None:
        """
        Load SpeechBrain ASR model. Only loads once if already loaded.

        Args:
            device: Device to load model on ('cpu' or 'cuda')
        """
        if self._loaded:
            return

        if self.model is None:
            print("Initializing Conformer ASR model...")
            self.model = SpeechBrainASR(
                model_path="speechbrain/asr-conformer-transformerlm-librispeech",
                save_dir="pretrained_models/asr-transformer-transformerlm-librispeech",
            )

        self.model.load(device=device)
        self._loaded = True

    def process_audio_file(self, audio_file: Path) -> Dict:
        """
        Process a single audio file and return transcription.

        Args:
            audio_file: Path to the audio file

        Returns:
            Dictionary with transcription results
        """
        try:
            if self.model is None or not self._loaded:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            # Get audio duration using audio processor
            duration = self.audio_processor.get_audio_duration(audio_file)

            # Transcribe using the ASR model
            transcription = self.model.transcribe(audio_file)

            result = {
                "file_name": audio_file.name,
                "transcription": transcription,
                "duration": duration,
                "status": "success",
            }

            return result

        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
            return {"file_name": audio_file.name, "error": str(e), "status": "error"}

    def cleanup(self) -> None:
        """Release resources."""
        if self.model:
            self.model.cleanup()

        # Clear memory using audio processor
        self.audio_processor.clear_memory()
        self._loaded = False


if __name__ == "__main__":
    import json

    # Hardcoded test parameters
    audio_file_path = "data/M18_05_01.wav"
    device = "cpu"

    # Initialize and run ASR processor
    processor = ASRProcessor()
    print(f"Loading ASR model on {device}...")
    processor.load_model(device=device)

    print(f"Transcribing {audio_file_path}...")
    result = processor.process_audio_file(Path(audio_file_path))

    # Print results
    print("\nResults:")
    print(json.dumps(result, indent=2))

    # Clean up
    processor.cleanup()
    print("Resources released.")
