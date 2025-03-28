import torchaudio
import torch
import gc
from pathlib import Path

from src.config.settings import settings


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
    def limit_cpu_cores(num_cores: int = None) -> None:
        """Limit the number of CPU cores used."""
        # Use default from MongoDB settings if not specified
        if num_cores is None:
            num_cores = settings.MONGODB_MIN_POOL_SIZE

        torch.set_num_threads(num_cores)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(num_cores)

    @staticmethod
    def clear_memory() -> None:
        """Clear unused memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def resample_audio(waveform, orig_sample_rate):
        """Resample audio to the standard sample rate from settings."""
        target_sample_rate = settings.AUDIO_SAMPLE_RATE

        if orig_sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=target_sample_rate
            )
            return resampler(waveform), target_sample_rate
        return waveform, orig_sample_rate

    @staticmethod
    def convert_to_mono(waveform):
        """Convert audio to mono if it has multiple channels."""
        if waveform.shape[0] > settings.AUDIO_CHANNELS:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    @staticmethod
    def normalize_audio(waveform, target_db=-20.0):
        """Normalize audio volume."""
        # Calculate current RMS (root mean square) value
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            # Calculate desired scale factor
            target_rms = 10 ** (target_db / 20)
            scale_factor = target_rms / rms
            # Scale the waveform
            return waveform * scale_factor
        return waveform
