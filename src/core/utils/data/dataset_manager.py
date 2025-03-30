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
        """
        If the dataset (the output subdirectory with audio files) already exists, skip the download.
        """
        try:
            # The desired output directory as specified in the settings
            output_dir = Path(settings.DATA_OUTPUT_DIR)
            # Our final expected location for the audio files is in an "output" subdirectory
            final_extracted_path = output_dir / "output"

            # Check if the dataset already exists (i.e., if the 'output' subdirectory exists and contains audio files)
            if final_extracted_path.exists():
                audio_files = list(final_extracted_path.rglob('*'))
                if any(file.suffix.lower() in settings.DATA_AUDIO_EXTENSIONS for file in audio_files):
                    print(f"Dataset already exists at {final_extracted_path}. Skipping download.")
                    self.dataset_path = output_dir
                    self.extracted_path = final_extracted_path
                    return output_dir

            # Create the output directory if it does not exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Download dataset from Kaggle
            # This is the kids speech dataset from the settings
            default_download_path_str = kagglehub.dataset_download(settings.DATA_KAGGLE_DATASET)
            default_download_path = Path(default_download_path_str)
            print("Dataset downloaded at default path:", default_download_path)

            # Check for the 'output' subdirectory in the default download location
            temp_output = default_download_path / "output"
            if temp_output.exists():
                # Move the entire 'output' folder to the final_extracted_path
                shutil.move(str(temp_output), str(final_extracted_path))
            else:
                # If there's no 'output' subdirectory, move all downloaded contents into final_extracted_path
                final_extracted_path.mkdir(parents=True, exist_ok=True)
                for item in default_download_path.iterdir():
                    shutil.move(str(item), str(final_extracted_path))

            # Set the internal paths for later use
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
        return any(file.suffix.lower() in settings.DATA_AUDIO_EXTENSIONS for file in audio_files)
