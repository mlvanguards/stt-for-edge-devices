import shutil
from pathlib import Path
from typing import Optional
import kagglehub
from src.config.settings import settings

class DatasetManager:
    def __init__(self):
        """Initialize the dataset manager with settings."""
        self.dataset_path: Optional[Path] = None
        self.extracted_path: Optional[Path] = None

    def download_dataset(self) -> Path:
        try:
            # Use the output directory defined in your settings (should be inside your repo)
            output_dir = Path(settings.data.DATA_OUTPUT_DIR)
            # Expect audio files to be inside an "output" subdirectory
            final_extracted_path = output_dir / "output"

            # If the dataset already exists here, skip download
            if final_extracted_path.exists():
                audio_files = list(final_extracted_path.rglob('*'))
                if any(file.suffix.lower() in settings.data.DATA_AUDIO_EXTENSIONS for file in audio_files):
                    print(f"Dataset already exists at {final_extracted_path}. Skipping download.")
                    self.dataset_path = output_dir
                    self.extracted_path = final_extracted_path
                    return output_dir

            # Create the output directory if it does not exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Download dataset from Kaggle using your dataset ID
            default_download_path_str = kagglehub.dataset_download(settings.data.DATA_KAGGLE_DATASET)
            default_download_path = Path(default_download_path_str)
            print("Dataset downloaded at default path:", default_download_path)

            # Check for the subdirectory containing audio files.
            # Try "output" first; if not found, check "extracted"
            temp_output = default_download_path / "output"
            if not temp_output.exists():
                temp_output = default_download_path / "extracted"

            if temp_output.exists():
                # Move the entire folder to the final destination in your repo
                shutil.move(str(temp_output), str(final_extracted_path))
            else:
                # Otherwise, move all downloaded contents into final_extracted_path
                final_extracted_path.mkdir(parents=True, exist_ok=True)
                for item in default_download_path.iterdir():
                    shutil.move(str(item), str(final_extracted_path))

            # Save internal paths for later use
            self.dataset_path = output_dir
            self.extracted_path = final_extracted_path
            return output_dir

        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            raise

    def get_dataset_path(self) -> Optional[Path]:
        """Return the path to the extracted dataset (i.e. the output subdirectory)."""
        return self.extracted_path

    def verify_dataset(self) -> bool:
        """Verify that the extracted dataset directory exists and contains at least one audio file."""
        if not self.extracted_path or not self.extracted_path.exists():
            return False

        # Check if at least one audio file exists in the extracted path
        audio_files = list(self.extracted_path.rglob('*'))
        return any(file.suffix.lower() in settings.data.DATA_AUDIO_EXTENSIONS for file in audio_files)
