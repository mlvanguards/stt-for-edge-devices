import torchaudio
import torch
import gc
from pathlib import Path

class AudioUtils:
    @staticmethod
    def get_audio_duration(audio_path: Path) -> float:
        """Get duration of audio file in seconds."""
        try:
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception as e:
            print(f"Error getting duration for {audio_path}: {str(e)}")
            return None

    @staticmethod
    def limit_cpu_cores(num_cores: int = 6) -> None:
        """Limit the number of CPU cores used."""
        torch.set_num_threads(num_cores)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(num_cores)

    @staticmethod
    def clear_memory() -> None:
        """Clear unused memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
