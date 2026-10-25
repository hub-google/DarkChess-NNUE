import os
import json
import gzip
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# Load environment variables from .env file
load_dotenv()

class HFClient:
    def __init__(self, repo_id="your-username/darkchess-training-data"):
        self.repo_id = repo_id
        self.token = os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)
        
        # Verify repo access
        try:
            self.api.dataset_info(self.repo_id)
        except RepositoryNotFoundError:
            print(f"Warning: Dataset repository '{self.repo_id}' not found or access denied.")
            print("Please create the dataset on Hugging Face and ensure you are logged in (huggingface-cli login) or have a valid HF_TOKEN in .env.")
        except Exception as e:
            print(f"Failed to authenticate or access Hugging Face: {e}")

    def list_training_files(self):
        """List all .jsonl.gz files in the Hugging Face dataset."""
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            return [f for f in files if f.endswith('.jsonl.gz')]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def download_and_parse(self, filename):
        """Download a compressed jsonl file from Hugging Face and parse it."""
        try:
            # Download file to local cache
            file_path = hf_hub_download(repo_id=self.repo_id, filename=filename, repo_type="dataset", token=self.token)
            
            data = []
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        except Exception as e:
            print(f"Error downloading or parsing {filename}: {e}")
            return []

# Example usage
if __name__ == '__main__':
    client = HFClient()
    files = client.list_training_files()
    print(f"Found {len(files)} training files.")
