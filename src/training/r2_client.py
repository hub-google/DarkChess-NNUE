import requests
import json
import os

class R2Client:
    def __init__(self, worker_url):
        self.worker_url = worker_url

    def list_training_files(self):
        try:
            response = requests.get(f"{self.worker_url}/list")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def download_file(self, filename, dest_dir="data"):
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        
        try:
            response = requests.get(f"{self.worker_url}/download/{filename}", stream=True)
            if response.status_code == 200:
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return dest_path
            return None
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None
