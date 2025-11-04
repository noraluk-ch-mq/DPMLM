import requests
import zipfile
from pathlib import Path
import random

class DatasetManager:
    def __init__(self, data_dir="data/uci_sentiment_dataset"):
        self.dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_uci_dataset(self):
        zip_path = self.data_dir / "sentiment_labelled_sentences.zip"
        extract_dir = self.data_dir / "extracted"

        if not zip_path.exists():
            try:
                print(f"   Downloading from: {self.dataset_url}")
                response = requests.get(self.dataset_url, stream=True)
                response.raise_for_status()

                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"   Download completed: {zip_path}")
            except Exception as e:
                print(f"   Download error: {e}")
                return False
        else:
            print(f"   File already exists: {zip_path}")

        if not extract_dir.exists():
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"   Extraction completed: {extract_dir}")
            except Exception as e:
                print(f"   Extraction error: {e}")
                return False
        else:
            print(f"   Files already extracted: {extract_dir}")

        return True

    def load_uci_data(self, filename):
        file_path = self.data_dir / "extracted" / "sentiment labelled sentences" / filename

        if not file_path.exists():
            print(f"   File not found: {file_path}")
            return []

        data = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.rsplit('\t', 1)
                        if len(parts) == 2:
                            text, label = parts
                            if text.strip() and label.strip() in ['0', '1']:
                                data.append({
                                    'text': text.strip(),
                                    'label': int(label.strip())
                                })
        except Exception as e:
            print(f"   Error reading file {filename}: {e}")

        return data

    def prepare_test_data(self):
        print("\nPreparing test data from UCI Sentiment Labeled Sentences Dataset...")

        if not self.download_uci_dataset():
            raise Exception("Failed to download UCI dataset")

        dataset_files = {
            'imdb': {
                'file': 'imdb_labelled.txt',
                'name': 'IMDB Movie Reviews',
                'description': 'Movie reviews from IMDB with sentiment labels'
            },
            'amazon': {
                'file': 'amazon_cells_labelled.txt',
                'name': 'Amazon Product Reviews',
                'description': 'Amazon product reviews with sentiment labels'
            },
            'yelp': {
                'file': 'yelp_labelled.txt',
                'name': 'Yelp Reviews',
                'description': 'Yelp business reviews with sentiment labels'
            }
        }

        datasets = {}
        total_samples = 0

        for dataset_key, dataset_info in dataset_files.items():
            print(f"\n   Loading {dataset_info['name']}...")
            data = self.load_uci_data(dataset_info['file'])

            if not data:
                print(f"      Unable to load data for {dataset_info['name']}")
                print(f"      Please check UCI dataset download")
                continue

            positive_samples = [item for item in data if item['label'] == 1]
            negative_samples = [item for item in data if item['label'] == 0]

            print(f"      Positive samples: {len(positive_samples)}")
            print(f"      Negative samples: {len(negative_samples)}")

            selected_samples = []

            if len(positive_samples) >= 4:
                selected_samples.extend(random.sample(positive_samples, 4))
            else:
                selected_samples.extend(positive_samples)

            if len(negative_samples) >= 4:
                selected_samples.extend(random.sample(negative_samples, 4))
            else:
                selected_samples.extend(negative_samples)

            datasets[dataset_key] = {
                'info': dataset_info,
                'samples': selected_samples
            }

            print(f"      Selected samples: {len(datasets[dataset_key]['samples'])} samples")
            total_samples += len(datasets[dataset_key]['samples'])

        if not datasets:
            raise Exception("Unable to load data from UCI dataset. Please check download")

        print(f"\nTest data preparation completed: {total_samples} total samples")
        return datasets