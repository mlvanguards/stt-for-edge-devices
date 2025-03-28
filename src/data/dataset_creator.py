from datasets import Dataset, DatasetDict, Audio
import os
import json

class DatasetCreator:
    @staticmethod
    def create_dataset(
            data_dir: str,
            audio_dir: str,
            config = None
    ) -> DatasetDict:
        """Create HuggingFace dataset from processed data."""
        splits = ['train', 'validation', 'test']
        data = {}

        for split in splits:
            json_path = os.path.join(data_dir, f"{split}.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                entries = json.load(f)

            for entry in entries:
                entry['audio'] = os.path.join(audio_dir, entry['file_name'])

            # Create a Dataset for this split
            data[split] = Dataset.from_dict({
                "file_name": [entry["file_name"] for entry in entries],
                "transcription": [entry["transcription"] for entry in entries],
                "duration": [entry["duration"] for entry in entries],
                "status": [entry["status"] for entry in entries],
                "audio": [entry["audio"] for entry in entries]
            }).cast_column("audio", Audio())

        return DatasetDict(data)

def main():
    data_dir = "split_data"
    audio_dir = "dataset_kaggle/output"
    dataset_dict = DatasetCreator.create_dataset(data_dir, audio_dir)
    dataset_dict.push_to_hub("StefanStefan/STT-test")

if __name__ == "__main__":
    main()
