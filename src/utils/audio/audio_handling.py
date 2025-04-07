# src/utils/audio/lightweight_audio_processor.py
import io
import wave
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
import os

if "PATH" not in os.environ:
    os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin"

from pydub import AudioSegment
from src.config.settings import settings

logger = logging.getLogger(__name__)

class AudioProcessorMainApp:
    """
    Audio processing class for the main FastAPI app.
    Contains only methods that do not rely on torch or torchaudio.
    """
    def __init__(self, config=None):
        self.config = config or settings
        self.sample_rate = self.config.audio.AUDIO_SAMPLE_RATE
        self.channels = self.config.audio.AUDIO_CHANNELS
        self.format = self.config.audio.AUDIO_FORMAT

    def get_audio_duration(self, audio_source: Union[str, Path, bytes]) -> float:
        try:
            if isinstance(audio_source, (str, Path)):
                audio = AudioSegment.from_file(str(audio_source))
            elif isinstance(audio_source, bytes):
                audio = AudioSegment.from_file(io.BytesIO(audio_source))
            else:
                raise ValueError(f"Unsupported type: {type(audio_source)}")
            return audio.duration_seconds
        except Exception as e:
            logger.error(f"Duration error: {e}")
            return 0.0

    def get_audio_format(self, content_type: str) -> str:
        format_map = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mpeg": "mp3",
            "audio/webm": "webm"
        }
        return format_map.get(content_type, "wav")

    def validate_content_type(self, content_type: str) -> bool:
        return content_type in self.config.audio.ALLOWED_AUDIO_CONTENT_TYPES

    def optimize_for_stt(self, audio_content: bytes, content_type: str) -> Optional[bytes]:
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_content),
                                           format=self.get_audio_format(content_type))
            optimized = audio.set_channels(self.channels).set_frame_rate(self.sample_rate)
            buffer = io.BytesIO()
            optimized.export(buffer, format=self.format)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Optimize error: {e}")
            return None

    def split_audio(self, audio_content: bytes, content_type: str,
                    segment_duration: Optional[float] = None) -> List[Dict]:
        if segment_duration is None:
            segment_duration = self.config.audio.AUDIO_SEGMENT_DURATION
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_content),
                                           format=self.get_audio_format(content_type))
            audio = audio.set_channels(self.channels).set_frame_rate(self.sample_rate)
            seg_ms = int(segment_duration * 1000)
            segments = []
            for i, start in enumerate(range(0, len(audio), seg_ms)):
                seg = audio[start:start+seg_ms]
                buffer = io.BytesIO()
                seg.export(buffer, format=self.format)
                segments.append({
                    "index": i,
                    "audio_bytes": buffer.getvalue(),
                    "start_time": start / 1000,
                    "end_time": min((start + seg_ms) / 1000, len(audio) / 1000)
                })
            return segments
        except Exception as e:
            logger.error(f"Split error: {e}")
            return []

    def create_silent_audio(self, duration: float = 0.5, sample_rate: Optional[int] = None) -> bytes:
        if sample_rate is None:
            sample_rate = self.sample_rate
        samples = np.zeros(int(sample_rate * duration), dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        return buffer.getvalue()
