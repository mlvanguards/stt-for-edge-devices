import json
import os
from sklearn.model_selection import train_test_split
from src.configurator.config import Config

class DataSplitter:
    @staticmethod
    def split_data(
        input_file: str,
        output_dir: str,
        config: Config
    ) -> None:
        """Split ASR data into training, validation and test sets."""
        os.makedirs(output_dir, exist_ok=True)

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading input file: {e}")
            return

        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data,
            test_size=config.processing['test_size'],
            random_state=config.processing['random_seed']
        )

        # Second split: separate train and validation from remaining data
        relative_val_size = config.processing['val_size'] / \
                          (config.processing['train_size'] + config.processing['val_size'])
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=relative_val_size,
            random_state=config.processing['random_seed']
        )

        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }

        for split_name, split_data in splits.items():
            output_file = os.path.join(output_dir, f'{split_name}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)