import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.audio_handling import AudioUtils
from src.config.settings import settings

class ASRProcessor:
    def __init__(self):
        """Initialize the ASR processor with settings."""
        self.model = None

    def load_model(self, device: str = None) -> None:
        """Load SpeechBrain ASR model."""
        device = device or settings.MODEL_DEVICE
        print("Loading Conformer ASR model...")
        from speechbrain.inference.ASR import EncoderDecoderASR
        self.model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-conformer-transformerlm-librispeech",
            savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
            run_opts={"device": device}
        )

    def process_audio_file(self, audio_file: Path) -> Dict:
        """Process a single audio file and return transcription."""
        try:
            transcription = self.model.transcribe_file(str(audio_file))
            duration = AudioUtils.get_audio_duration(audio_file)

            result = {
                "file_name": audio_file.name,
                "transcription": transcription,
                "duration": duration,
                "status": "success"
            }

            AudioUtils.clear_memory()
            return result

        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
            return {
                "file_name": audio_file.name,
                "error": str(e),
                "status": "error"
            }

    def process_dataset(
            self,
            input_dir: str,
            output_file: str,
            max_duration: Optional[float] = None,
            batch_size: Optional[int] = None,
            save_interval: Optional[int] = None
    ) -> List[Dict]:
        """Process audio files with duration filtering."""
        max_duration = max_duration or settings.DATA_MAX_AUDIO_DURATION
        batch_size = batch_size or settings.MODEL_BATCH_SIZE
        save_interval = save_interval or settings.MODEL_SAVE_INTERVAL

        # Set CPU core limit
        AudioUtils.limit_cpu_cores(settings.MODEL_NUM_CORES)

        input_path = Path(input_dir)
        potential_subdir = input_path / "output"
        if potential_subdir.exists() and potential_subdir.is_dir():
            input_path = potential_subdir

        # Search for audio files recursively in the determined input_path
        audio_files = []
        for ext in settings.DATA_AUDIO_EXTENSIONS:
            audio_files.extend(input_path.rglob(f"*{ext}"))
        print(f"Found {len(audio_files)} audio files in directory {input_path}")

        # Filter files based on duration
        filtered_files = []
        for file in audio_files:
            duration = AudioUtils.get_audio_duration(file)
            if duration is not None and duration <= max_duration:
                filtered_files.append(file)

        print(f"\nFound {len(filtered_files)}/{len(audio_files)} files under {max_duration} seconds")

        results = []
        files_to_process = []

        # Load existing results if any
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_files = {r['file_name'] for r in results}
            files_to_process = [f for f in filtered_files if f.name not in processed_files]
            print(f"Will process {len(files_to_process)} new files")
        else:
            files_to_process = filtered_files
            print(f"No previous results found. Will process all {len(files_to_process)} files")

        if not files_to_process:
            print("No files to process!")
            return results

        # Process files in batches
        for i, audio_file in enumerate(files_to_process):
            result = self.process_audio_file(audio_file)
            results.append(result)

            if (i + 1) % batch_size == 0:
                AudioUtils.clear_memory()
                time.sleep(0.1)

            if (i + 1) % save_interval == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nIntermediate results saved after {len(results)} files")

        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nAll results saved: {len(results)} files processed")

        return results
