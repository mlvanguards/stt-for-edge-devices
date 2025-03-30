from src.asr.base import ASRModelInterface
from src.asr.speech_brain import SpeechBrainASR
from src.config import settings


class ASRModelFactory:
    """Factory for creating ASR model instances."""

    @staticmethod
    def create_model(model_type: str, config: settings) -> ASRModelInterface:
        """Create an ASR model based on the specified type."""
        if model_type.lower() == "speechbrain":
            return SpeechBrainASR(
                model_path=config.model.get(
                    "path", "speechbrain/asr-conformer-transformerlm-librispeech"
                ),
                save_dir=config.model.get(
                    "save_dir",
                    "pretrained_models/asr-transformer-transformerlm-librispeech",
                ),
            )
        # Add more model types as needed
        else:
            raise ValueError(f"Unsupported ASR model type: {model_type}")
