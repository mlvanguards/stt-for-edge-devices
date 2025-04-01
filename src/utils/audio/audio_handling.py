import io
import wave
import logging
import numpy as np
import torch
import torchaudio
import gc
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from pydub import AudioSegment

from src.config.settings import settings

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Centralized audio processing utility that handles all audio-related operations
    including loading, conversion, normalization, and optimization for speech recognition.

    This class consolidates functionality previously spread across:
    - src/core/utils/audio/audio_handling.py
    - src/asr/speech_brain.py
    - src/resource_testing/stt_edge_profiler.py
    """

    def __init__(self, config=None):
        """
        Initialize the audio processor with application configuration.

        Args:
            config: Optional configuration (uses settings by default)
        """
        self.config = config or settings
        self.sample_rate = self.config.audio.AUDIO_SAMPLE_RATE
        self.channels = self.config.audio.AUDIO_CHANNELS
        self.format = self.config.audio.AUDIO_FORMAT

    def get_audio_duration(self, audio_source: Union[str, Path, bytes]) -> float:
        """
        Get duration of audio in seconds.

        Args:
            audio_source: Path to audio file or audio bytes

        Returns:
            Duration in seconds or None if failed
        """
        try:
            if isinstance(audio_source, (str, Path)):
                info = torchaudio.info(audio_source)
                return info.num_frames / info.sample_rate
            elif isinstance(audio_source, bytes):
                # Handle audio bytes
                with io.BytesIO(audio_source) as buffer:
                    with wave.open(buffer, 'rb') as wav:
                        return wav.getnframes() / wav.getframerate()
            else:
                raise ValueError(f"Unsupported audio source type: {type(audio_source)}")
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return None

    def load_audio(self,
                   audio_source: Union[str, Path, bytes],
                   normalize: bool = True) -> Tuple[torch.Tensor, int]:
        """
        Load audio from file, path, or bytes with standardized processing.

        Args:
            audio_source: Audio file path or bytes
            normalize: Whether to normalize audio volume

        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            if isinstance(audio_source, (str, Path)):
                # Load from file
                waveform, sample_rate = torchaudio.load(audio_source)
            elif isinstance(audio_source, bytes):
                # Load from bytes
                with io.BytesIO(audio_source) as buffer:
                    waveform, sample_rate = torchaudio.load(buffer)
            else:
                raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

            # Convert to mono if needed
            if waveform.shape[0] > self.channels:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != self.sample_rate:
                waveform, sample_rate = self.resample(waveform, sample_rate, self.sample_rate)

            # Normalize if requested
            if normalize:
                waveform = self.normalize_audio(waveform)

            return waveform, sample_rate

        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise

    def resample(self,
                 waveform: torch.Tensor,
                 orig_sample_rate: int,
                 target_sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Resample audio to the target sample rate.

        Args:
            waveform: Audio waveform
            orig_sample_rate: Original sample rate
            target_sample_rate: Target sample rate (uses settings if None)

        Returns:
            Tuple of (resampled_waveform, new_sample_rate)
        """
        if target_sample_rate is None:
            target_sample_rate = self.sample_rate

        if orig_sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=target_sample_rate
            )
            return resampler(waveform), target_sample_rate
        return waveform, orig_sample_rate

    def normalize_audio(self,
                        waveform: torch.Tensor,
                        target_db: float = -20.0) -> torch.Tensor:
        """
        Normalize audio volume to a target dB level.

        Args:
            waveform: Audio waveform
            target_db: Target decibel level

        Returns:
            Normalized waveform
        """
        # Calculate current RMS (root mean square) value
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            # Calculate desired scale factor
            target_rms = 10 ** (target_db / 20)
            scale_factor = target_rms / rms
            # Scale the waveform
            return waveform * scale_factor
        return waveform

    def convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mono if it has multiple channels.

        Args:
            waveform: Audio waveform

        Returns:
            Mono waveform
        """
        if waveform.shape[0] > self.channels:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def get_audio_format(self, content_type: str) -> str:
        """
        Convert MIME type to audio format.

        Args:
            content_type: MIME type

        Returns:
            Audio format string
        """
        format_map = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mpeg": "mp3",
            "audio/webm": "webm"
        }
        return format_map.get(content_type, "wav")

    def validate_content_type(self, content_type: str) -> bool:
        """
        Validate file content type against allowed types.

        Args:
            content_type: MIME type

        Returns:
            True if valid, False otherwise
        """
        return content_type in self.config.audio.ALLOWED_AUDIO_CONTENT_TYPES

    def optimize_for_stt(self,
                         audio_content: bytes,
                         content_type: str) -> bytes:
        """
        Optimize audio for speech recognition.
        Returns optimized audio bytes in WAV format.

        Args:
            audio_content: Raw audio bytes
            content_type: MIME type

        Returns:
            Optimized audio bytes or None if failed
        """
        try:
            # Load audio from bytes
            audio_format = self.get_audio_format(content_type)
            audio = AudioSegment.from_file(io.BytesIO(audio_content), format=audio_format)

            # Optimize: Convert to mono, set sample rate
            optimized = audio.set_channels(self.channels).set_frame_rate(self.sample_rate)

            # Export to WAV format
            buffer = io.BytesIO()
            optimized.export(buffer, format=self.format)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error optimizing audio: {str(e)}")
            return None

    def split_audio(self,
                    audio_content: bytes,
                    content_type: str,
                    segment_duration: Optional[float] = None) -> List[Dict]:
        """
        Split audio into segments using pydub.
        Returns a list of segment dictionaries with audio and metadata.

        Args:
            audio_content: Raw audio bytes
            content_type: MIME type
            segment_duration: Segment duration in seconds

        Returns:
            List of segment dictionaries with audio and metadata
        """
        # Use settings value or default
        if segment_duration is None:
            segment_duration = self.config.audio.AUDIO_SEGMENT_DURATION

        try:
            # Load audio from bytes
            audio_format = self.get_audio_format(content_type)
            audio = AudioSegment.from_file(io.BytesIO(audio_content), format=audio_format)

            # Optimize audio first
            audio = audio.set_channels(self.channels).set_frame_rate(self.sample_rate)

            # Split the audio into segments
            segment_length_ms = int(segment_duration * 1000)
            segments = []

            for i, start_ms in enumerate(range(0, len(audio), segment_length_ms)):
                # Extract segment
                segment = audio[start_ms:start_ms + segment_length_ms]

                # Export to WAV format
                buffer = io.BytesIO()
                segment.export(buffer, format=self.format)

                segments.append({
                    "index": i,
                    "audio_bytes": buffer.getvalue(),
                    "start_time": start_ms / 1000,
                    "end_time": min((start_ms + segment_length_ms) / 1000, len(audio) / 1000)
                })

            return segments
        except Exception as e:
            logger.error(f"Error splitting audio: {str(e)}")
            return []

    def create_silent_audio(self,
                            duration: float = 0.5,
                            sample_rate: Optional[int] = None) -> bytes:
        """
        Create silent audio for testing or API warming.

        Args:
            duration: Duration in seconds
            sample_rate: Sample rate (uses settings if None)

        Returns:
            Silent audio bytes
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        samples = np.zeros(int(sample_rate * duration), dtype=np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

        return buffer.getvalue()

    @staticmethod
    def clear_memory() -> None:
        """Clear unused memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def limit_cpu_cores(num_cores: int = None) -> None:
        """
        Limit the number of CPU cores used.

        Args:
            num_cores: Number of cores to use
        """
        if num_cores is None:
            return

        torch.set_num_threads(num_cores)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(num_cores)
