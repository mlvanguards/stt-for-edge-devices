import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

# Configure logging
logger = logging.getLogger(__name__)


class ASRModelInterface(ABC):
    """Abstract interface for ASR models."""

    @abstractmethod
    def load(self, device: str) -> None:
        """Load the ASR model."""
        pass

    @abstractmethod
    def transcribe(self, audio_file: Union[str, Path]) -> str:
        """Transcribe an audio file."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources."""
        pass
