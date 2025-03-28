from src.data.dataset_manager import DatasetManager
from src.asr.asr_processor import ASRProcessor
from src.data.data_normalizer import DataNormalizer
from src.data.data_splitter import DataSplitter
from src.data.dataset_creator import DatasetCreator
from src.config.settings import settings

def main():
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    dataset_path = dataset_manager.download_dataset()

    # Verify dataset
    if not dataset_manager.verify_dataset():
        print("Error: Dataset verification failed")
        return

    # Process dataset
    processor = ASRProcessor()
    processor.load_model()

    # Process audio files
    results = processor.process_dataset(
        input_dir=str(dataset_path),
        output_file="../data/transcriptions-test.json",
        max_duration=settings.DATA_MAX_AUDIO_DURATION,
        batch_size=settings.MODEL_BATCH_SIZE,
        save_interval=settings.MODEL_SAVE_INTERVAL
    )

    # Normalize data
    DataNormalizer.process_data(
        input_file="../data/transcriptions-test.json",
        output_dir="processed_data"
    )

    # Split data
    DataSplitter.split_data(
        input_file="processed_data/normalized_clean_asr.json",
        output_dir="split_data"
    )

    # Create dataset
    dataset_creator = DatasetCreator()
    dataset = dataset_creator.create_dataset(
        data_dir="split_data",
        audio_dir=str(dataset_manager.get_dataset_path())
    )

    # Push to hub
    dataset.push_to_hub("StefanStefan/STT-test")


if __name__ == "__main__":
    main()
