import json
import os
from sklearn.model_selection import train_test_split

from src.config.settings import settings


class DataSplitter:
    @staticmethod
    def split_data(
            input_file: str,
            output_dir: str,
            config=None  # Kept for backward compatibility but not used
    ) -> None:
        """Split ASR data into training, validation and test sets using settings."""
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
            test_size=settings.DATA_TEST_SIZE,
            random_state=settings.DATA_RANDOM_SEED
        )

        # Second split: separate train and validation from remaining data
        relative_val_size = settings.DATA_VAL_SIZE / \
                            (settings.DATA_TRAIN_SIZE + settings.DATA_VAL_SIZE)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=relative_val_size,
            random_state=settings.DATA_RANDOM_SEED
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

        print(f"\nData splitting complete:")
        print(f"- Total samples: {len(data)}")
        print(f"- Training samples: {len(train_data)} ({len(train_data) / len(data) * 100:.1f}%)")
        print(f"- Validation samples: {len(val_data)} ({len(val_data) / len(data) * 100:.1f}%)")
        print(f"- Test samples: {len(test_data)} ({len(test_data) / len(data) * 100:.1f}%)")
