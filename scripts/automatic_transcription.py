from src.utils.data.dataset_manager import DatasetManager
from src.asr.asr_processor import ASRProcessor
from src.utils.data.data_normalizer import DataNormalizer
from src.utils.data.data_splitter import DataSplitter
from src.utils.data.dataset_creator import DatasetCreator
from src.config.settings import settings
from src.utils.audio.audio_handling import AudioProcessor
from src.asr.dataset_processor import DatasetProcessor


def main():
    # Initialize audio processor
    audio_processor = AudioProcessor()

    # Initialize dataset manager
    dataset_manager = DatasetManager()
    dataset_path = dataset_manager.download_dataset()

    # Verify dataset
    if not dataset_manager.verify_dataset():
        print("Error: Dataset verification failed")
        return

    # Initialize ASR processor with audio processor
    asr_processor = ASRProcessor(audio_processor=audio_processor)
    asr_processor.load_model()

    # Create dataset processor
    dataset_processor = DatasetProcessor(
        asr_processor=asr_processor,
        config=settings,
        audio_processor=audio_processor
    )

    # Process audio files
    results = dataset_processor.process_dataset(
        input_dir=str(dataset_path),
        output_file="../../data/transcriptions-test.json",
        max_duration=settings.data.DATA_MAX_AUDIO_DURATION,
        batch_size=settings.model.MODEL_BATCH_SIZE,
        save_interval=settings.model.MODEL_SAVE_INTERVAL
    )

    # Normalize data
    DataNormalizer.process_data(
        input_file="../../data/transcriptions-test.json",
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
