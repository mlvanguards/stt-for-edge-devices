from src.data_config.config import Config
from src.data.dataset_manager import DatasetManager
from src.asr.asr_processor import ASRProcessor
from src.data.data_normalizer import DataNormalizer
from src.data.data_splitter import DataSplitter
from src.data.dataset_creator import DatasetCreator


def main():
    # Initialize configuration
    config = Config()
    # Download dataset
    dataset_manager = DatasetManager(config)
    dataset_path = dataset_manager.download_dataset()

    # Verify dataset
    if not dataset_manager.verify_dataset():
        print("Error: Dataset verification failed")
        return

    # Process dataset
    processor = ASRProcessor(config)
    processor.load_model()

    # Process audio files
    results = processor.process_dataset(
        input_dir=str(dataset_path),
        output_file="../data/transcriptions-test.json",
        max_duration=config.dataset['max_duration'],
        batch_size=config.model['batch_size'],
        save_interval=config.model['save_interval']
    )

    # Normalize data
    DataNormalizer.process_data(
        input_file="../data/transcriptions-test.json",
        output_dir="processed_data",
        config=config
    )

    # Split data
    DataSplitter.split_data(
        input_file="processed_data/normalized_clean_asr.json",
        output_dir="split_data",
        config=config
    )

    # Create dataset
    dataset_creator = DatasetCreator()
    dataset = dataset_creator.create_dataset(
        data_dir="split_data",
        audio_dir=str(dataset_manager.get_dataset_path()),
        config=config
    )

    # Push to hub
    dataset.push_to_hub("StefanStefan/STT-test")


if __name__ == "__main__":
    main()