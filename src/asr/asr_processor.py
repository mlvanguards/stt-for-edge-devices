from pathlib import Path
from typing import Dict

from src.asr.base import ASRModelInterface
from src.asr.speech_brain import SpeechBrainASR
from src.core.utils.audio.audio_handling import AudioUtils


class ASRProcessor:
    def __init__(self, model: ASRModelInterface = None):
        """Initialize the ASR processor with settings."""
        self.model = model

    def load_model(self, device: str = "cpu") -> None:
        """Load SpeechBrain ASR model."""
        if self.model is None:
            print("Initializing Conformer ASR model...")
            self.model = SpeechBrainASR(
                model_path="speechbrain/asr-conformer-transformerlm-librispeech",
                save_dir="pretrained_models/asr-transformer-transformerlm-librispeech",
            )

        self.model.load(device=device)

    def process_audio_file(self, audio_file: Path) -> Dict:
        """Process a single audio file and return transcription."""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            transcription = self.model.transcribe(audio_file)
            duration = AudioUtils.get_audio_duration(audio_file)

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


if __name__ == "__main__":
    import json

    # Hardcoded test parameters
    audio_file_path = "/Users/vesaalexandru/Workspaces/cube/stt-for-edge-devices/data/M18_05_01.wav"  # Replace with your test audio file path
    device = "cpu"  # Use "cuda" if you have GPU support

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
