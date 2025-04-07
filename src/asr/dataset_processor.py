import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import torchaudio

from src.asr.asr_processor import ASRProcessor
from src.config.settings import settings
from src.utils.audio.audio_process import AudioProcessor

# Configure logging
logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Handles processing of audio datasets with ASR.
    Uses the centralized AudioProcessor for all audio operations.
    """

    def __init__(self, asr_processor: ASRProcessor, config: settings, audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize the dataset processor.

        Args:
            asr_processor: The ASR processor to use for transcription
            config: Configuration object containing dataset settings
            audio_processor: Optional audio processor instance
        """
        self.asr_processor = asr_processor
        self.config = config
        self.audio_processor = audio_processor or AudioProcessor(config)

    def find_audio_files(self, input_dir: str) -> List[Path]:
        """
        Find all audio files in the input directory.

        Args:
            input_dir: Directory to search for audio files

        Returns:
            List of paths to audio files
        """
        # Determine input directory
        input_path = Path(input_dir)
        potential_subdir = input_path / "output"
        if potential_subdir.exists() and potential_subdir.is_dir():
            input_path = potential_subdir
            logger.info(f"Using subdirectory: {input_path}")

        # Find audio files
        audio_extensions = self.config.data.DATA_AUDIO_EXTENSIONS
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.rglob(f"*{ext}"))

        logger.info(f"Found {len(audio_files)} audio files in directory {input_path}")
        return audio_files

    def filter_by_duration(
        self, audio_files: List[Path], max_duration: float
    ) -> List[Path]:
        """
        Filter audio files by duration using AudioProcessor.

        Args:
            audio_files: List of audio file paths
            max_duration: Maximum duration in seconds

        Returns:
            Filtered list of audio file paths
        """
        filtered_files = []
        for file in audio_files:
            try:
                # Use AudioProcessor to get duration
                duration = self.audio_processor.get_audio_duration(file)
                if duration is not None and duration <= max_duration:
                    filtered_files.append(file)
            except Exception as e:
                logger.warning(f"Could not determine duration of {file.name}: {str(e)}")

        logger.info(
            f"Found {len(filtered_files)}/{len(audio_files)} files under {max_duration} seconds"
        )
        return filtered_files

    def load_existing_results(
        self, output_file: str
    ) -> tuple[List[Dict[str, Any]], Set[str]]:
        """
        Load existing results from output file.

        Args:
            output_file: Path to the output file

        Returns:
            Tuple of (results list, set of processed file names)
        """
        results = []
        processed_files = set()

        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                processed_files = {r["file_name"] for r in results}
                logger.info(f"Loaded {len(results)} existing results")
            except Exception as e:
                logger.error(f"Error loading existing results: {str(e)}")
        else:
            logger.info("No previous results found")

        return results, processed_files

    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save results to output file.

        Args:
            results: List of transcription results
            output_file: Path to the output file
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(results)} results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def resample_audio_file(self, audio_file: Path, target_sample_rate: int = 16000,
                            output_dir: Optional[Path] = None) -> Path:
        """
        Resample the given audio file to the target sample rate (default 16kHz)
        and save the result in an output directory.
        """
        if output_dir is None:
            output_dir = audio_file.parent / "resampled"
            output_dir.mkdir(exist_ok=True)

        # Load audio without normalizing (so we work with raw waveform)
        waveform, sample_rate = self.audio_processor.load_audio(audio_file, normalize=False)

        # If the audio is not already 16kHz, resample it
        if sample_rate != target_sample_rate:
            waveform, _ = self.audio_processor.resample(waveform, sample_rate, target_sample_rate)
        else:
            pass

        # Define the output file path (using the same file name)
        output_path = output_dir / audio_file.name

        # Save the resampled waveform as a WAV file
        torchaudio.save(str(output_path), waveform, target_sample_rate)

        return output_path

    def process_dataset(
        self,
        input_dir: str,
        output_file: str,
        max_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        save_interval: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process audio files with duration filtering.

        Args:
            input_dir: Directory containing audio files
            output_file: Path to save transcription results
            max_duration: Maximum audio duration in seconds
            batch_size: Number of files to process before clearing memory
            save_interval: Number of files to process before saving results

        Returns:
            List of transcription results as dictionaries
        """
        # Get configuration parameters with fallbacks
        max_duration = max_duration or self.config.data.DATA_MAX_AUDIO_DURATION
        batch_size = batch_size or self.config.model.MODEL_BATCH_SIZE
        save_interval = save_interval or self.config.model.MODEL_SAVE_INTERVAL

        # Set CPU core limit if specified
        if "num_cores" in self.config.model:
            self.audio_processor.limit_cpu_cores(self.config.model.MODEL_NUM_CORES)

        # Find and filter audio files
        audio_files = self.find_audio_files(input_dir)
        filtered_files = self.filter_by_duration(audio_files, max_duration)

        # Load existing results if any
        results, processed_files = self.load_existing_results(output_file)
        files_to_process = [f for f in filtered_files if f.name not in processed_files]

        logger.info(f"Will process {len(files_to_process)} new files")

        if not files_to_process:
            logger.info("No files to process!")
            return results

        # Process files in batches
        total_files = len(files_to_process)
        for i, audio_file in enumerate(files_to_process):
            # Resample the audio file to 16kHz and get the new file path
            resampled_file = self.resample_audio_file(audio_file, target_sample_rate=16000)

            # Now process the resampled file for transcription
            result = self.asr_processor.process_audio_file(resampled_file)
            results.append(result)

            # Log progress
            if result["status"] == "success":
                logger.info(
                    f"[{i + 1}/{total_files}] Processed {result['file_name']} ({result['duration']:.2f}s)"
                )
            else:
                logger.error(
                    f"[{i + 1}/{total_files}] Failed to process {result['file_name']}: {result.get('error', 'Unknown error')}"
                )

            # Clear memory after each batch
            if (i + 1) % batch_size == 0:
                self.asr_processor.cleanup()
                self.asr_processor.load_model()  # Reload the model
                time.sleep(0.1)

            # Save intermediate results
            if (i + 1) % save_interval == 0 or i == total_files - 1:
                self.save_results(results, output_file)

        return results
