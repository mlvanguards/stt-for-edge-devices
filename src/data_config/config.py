class Config:
    def __init__(self):
        # Dataset settings
        self.dataset = {
            'kaggle_dataset': 'mirfan899/kids-speech-dataset',
            'output_dir': 'dataset_kaggle',
            'audio_extensions': ['.wav', '.flac', '.mp3'],
            'max_duration': 10
        }

        # Model settings
        self.model = {
            'device': 'cpu',
            'num_cores': 6,
            'batch_size': 10,
            'save_interval': 50
        }

        # Processing settings
        self.processing = {
            'train_size': 0.7,
            'val_size': 0.15,
            'test_size': 0.15,
            'random_seed': 42
        }