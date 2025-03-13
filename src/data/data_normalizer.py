import json
import re
from collections import Counter
import os
from typing import Optional
from src.data_config.config import Config

class DataNormalizer:
    @staticmethod
    def normalize_transcription(text: str) -> str:
        """Normalize transcription text for ASR training."""
        # Convert to lowercase
        text = text.lower()

        # Standardize apostrophes
        text = text.replace("'", "'")

        # Remove special characters except apostrophes
        text = re.sub(r'[^a-z0-9\'\s]', '', text)

        # Remove multiple spaces and strip
        return ' '.join(text.split())

    @staticmethod
    def detect_likely_errors(text: str) -> bool:
        """Detect likely transcription errors based on various heuristics."""
        words = text.split()

        # Check 1: No repeated words in sequence
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                return False

        # Check 2: Word frequency
        word_counts = Counter(words)
        for count in word_counts.values():
            if count > 3:  # Same word appears more than 3 times
                return False

        # Check 3: Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        if not (2 <= avg_word_length <= 10):
            return False

        # Check 4: Percentage of short words
        short_words = sum(1 for word in words if len(word) == 1)
        if short_words / len(words) > 0.2:  # More than 20% are single letters
            return False

        return True

    @staticmethod
    def process_data(
        input_file: str,
        output_dir: str = "processed_data",
        config: Optional[Config] = None
    ) -> None:
        """Process ASR dataset and save normalized/filtered results."""
        os.makedirs(output_dir, exist_ok=True)

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading input file: {e}")
            return

        processed_data = []
        filtered_data = []

        for item in data:
            try:
                if 'transcription' not in item or 'file_name' not in item:
                    print(f"Skipping item due to missing required fields: {item}")
                    continue

                normalized_text = DataNormalizer.normalize_transcription(item['transcription'])
                processed_item = {
                    'file_name': item['file_name'],
                    'transcription': normalized_text,
                    'duration': item.get('duration', None),
                    'status': item.get('status', None)
                }

                if DataNormalizer.detect_likely_errors(normalized_text):
                    processed_data.append(processed_item)
                else:
                    filtered_data.append(processed_item)

            except Exception as e:
                print(f"Error processing item {item}: {e}")
                continue

        # Save processed data
        processed_path = os.path.join(output_dir, "normalized_clean_asr.json")
        filtered_path = os.path.join(output_dir, "normalized_filtered_asr.json")

        try:
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            with open(filtered_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving output files: {e}")
            return

        print(f"\nProcessing complete:")
        print(f"- Original samples: {len(data)}")
        print(f"- Clean samples: {len(processed_data)}")
        print(f"- Filtered samples: {len(filtered_data)}")