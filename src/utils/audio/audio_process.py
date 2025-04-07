import io
import logging
import torch
import torchaudio
import gc
from pathlib import Path
from typing import Optional, Tuple, Union
from src.utils.audio.audio_handling import AudioProcessorMainApp

logger = logging.getLogger(__name__)

class AudioProcessor(AudioProcessorMainApp):
    """
    Audio processing class that includes the additional methods
    requiring torch and torchaudio. Inherits from LightweightAudioProcessor.
    """
    def load_audio(self, audio_source: Union[str, Path, bytes], normalize: bool = True) -> Tuple[torch.Tensor, int]:
        try:
            if isinstance(audio_source, (str, Path)):
                waveform, sr = torchaudio.load(audio_source)
            elif isinstance(audio_source, bytes):
                with io.BytesIO(audio_source) as buf:
                    waveform, sr = torchaudio.load(buf)
            else:
                raise ValueError(f"Unsupported type: {type(audio_source)}")
            if waveform.shape[0] > self.channels:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform, sr = self.resample(waveform, sr, self.sample_rate)
            if normalize:
                waveform = self.normalize_audio(waveform)
            return waveform, sr
        except Exception as e:
            logger.error(f"Load audio error: {e}")
            raise

    def resample(self, waveform: torch.Tensor, orig_sr: int, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        if target_sr is None:
            target_sr = self.sample_rate
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            return resampler(waveform), target_sr
        return waveform, orig_sr

    def normalize_audio(self, waveform: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            return waveform * (target_rms / rms)
        return waveform

    def convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] > self.channels:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    @staticmethod
    def clear_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def limit_cpu_cores(num_cores: int = None) -> None:
        if num_cores:
            torch.set_num_threads(num_cores)
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(num_cores)
