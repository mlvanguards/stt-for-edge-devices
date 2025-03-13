import io
import logging
from pydub import AudioSegment
from src.config.settings import (
    ALLOWED_AUDIO_CONTENT_TYPES,
    AUDIO_SEGMENT_DURATION
)

logger = logging.getLogger(__name__)


def check_ffmpeg():
    """
    Check if audio processing is available
    For serverless, we use pydub which doesn't need system ffmpeg
    """
    try:
        # Just create a simple silent segment to test pydub
        AudioSegment.silent(duration=1)
        return True
    except Exception as e:
        logger.error(f"Audio processing unavailable: {str(e)}")
        return False


def validate_content_type(content_type):
    """Validate file content type"""
    return content_type in ALLOWED_AUDIO_CONTENT_TYPES


def get_audio_format(content_type):
    """Convert MIME type to audio format"""
    format_map = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/webm": "webm"
    }
    return format_map.get(content_type, "wav")


def optimize_audio_for_stt(audio_content, content_type):
    """
    Optimize audio for speech recognition using pydub
    Returns optimized audio bytes and format
    """
    try:
        # Load audio from bytes
        audio_format = get_audio_format(content_type)
        audio = AudioSegment.from_file(io.BytesIO(audio_content), format=audio_format)

        # Optimize: Convert to mono, set sample rate to 16kHz
        optimized = audio.set_channels(1).set_frame_rate(16000)

        # Export to WAV format
        buffer = io.BytesIO()
        optimized.export(buffer, format="wav")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error optimizing audio: {str(e)}")
        return None


def split_audio_file(audio_content, content_type, segment_duration=AUDIO_SEGMENT_DURATION):
    """
    Split audio into segments using pydub
    Returns a list of audio segment bytes
    """
    try:
        # Load audio from bytes
        audio_format = get_audio_format(content_type)
        audio = AudioSegment.from_file(io.BytesIO(audio_content), format=audio_format)

        # Optimize audio first
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Split the audio into segments
        segment_length_ms = segment_duration * 1000
        segments = []

        for i, start_ms in enumerate(range(0, len(audio), segment_length_ms)):
            # Extract segment
            segment = audio[start_ms:start_ms + segment_length_ms]

            # Export to WAV format
            buffer = io.BytesIO()
            segment.export(buffer, format="wav")

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