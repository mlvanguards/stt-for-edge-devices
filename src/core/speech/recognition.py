import time
import logging
import requests
import re

from src.config.settings import (
    HUGGINGFACE_TOKEN,
    HUGGINGFACE_API_URL,
    SPEECH_RECOGNITION_RETRIES,
    SPEECH_RECOGNITION_BACKOFF_FACTOR
)

logger = logging.getLogger(__name__)


def process_audio_file(audio_content, content_type, force_split=False):
    """
    Process an audio file by sending it directly to HuggingFace API.
    No preprocessing or ffmpeg/pydub dependencies.

    Args:
        audio_content: Raw audio bytes
        content_type: MIME type of the audio
        force_split: Ignored parameter (kept for API compatibility)

    Returns:
        List of transcription segments (typically just one)
    """
    try:
        logger.info(f"Processing audio directly via HuggingFace (content type: {content_type})")

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": content_type,
        }

        # Try multiple times with exponential backoff
        for attempt in range(SPEECH_RECOGNITION_RETRIES):
            try:
                response = requests.post(HUGGINGFACE_API_URL, headers=headers, data=audio_content)

                if response.status_code == 200:
                    transcription_result = response.json()

                    # Extract the text from the transcription
                    if isinstance(transcription_result, dict) and "text" in transcription_result:
                        text = transcription_result["text"]
                        if text and "[Failed to transcribe" not in text:
                            return [{"index": 0, "text": text}]
                    else:
                        text = str(transcription_result)
                        if text and "[Failed to transcribe" not in text:
                            return [{"index": 0, "text": text}]

                elif response.status_code == 503:
                    # Service unavailable, wait and retry
                    wait_time = (SPEECH_RECOGNITION_BACKOFF_FACTOR ** attempt)
                    logger.warning(f"API returned 503, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Other error
                    logger.error(f"API error: {response.status_code} - {response.text}")

                    if attempt < SPEECH_RECOGNITION_RETRIES - 1:
                        # Try again
                        time.sleep(1)
                    else:
                        # Last attempt failed
                        return [{"index": 0, "text": f"Failed to transcribe audio: API error {response.status_code}"}]

            except Exception as e:
                logger.error(f"Error in processing attempt {attempt + 1}: {str(e)}")

                if attempt < SPEECH_RECOGNITION_RETRIES - 1:
                    # Try again
                    time.sleep(1)
                else:
                    # Last attempt failed
                    return [{"index": 0, "text": f"Error processing audio: {str(e)}"}]

        # If we're here, all attempts failed
        return [{"index": 0, "text": "Failed to transcribe audio after multiple attempts"}]

    except Exception as e:
        logger.error(f"Unexpected error in audio processing: {str(e)}")
        return [{"index": 0, "text": "Error processing audio file."}]


def clean_transcription(transcriptions):
    """
    Process and clean transcription results.
    Returns cleaned transcription text.
    """
    # Sort transcriptions by index
    sorted_transcriptions = sorted(transcriptions, key=lambda x: x.get("index", 0))

    # Combine all transcriptions
    full_text = " ".join([t.get("text", "") for t in sorted_transcriptions])

    # Clean up the transcription text if it contains failure messages
    clean_text = re.sub(r'\[Segment \d+ transcription failed\]\s*', '', full_text).strip()
    clean_text = re.sub(r'Failed to transcribe audio.*$', '', clean_text).strip()
    clean_text = re.sub(r'Error processing audio.*$', '', clean_text).strip()

    # If nothing is left after cleaning, return a default message
    if not clean_text:
        clean_text = "Unable to transcribe audio clearly. Please try again with a clearer recording."

    return clean_text