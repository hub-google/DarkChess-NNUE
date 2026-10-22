import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from hf_bootstrap import ensure_dataset
from huggingface_hub.errors import RepositoryNotFoundError


class HFBootstrapTests(unittest.TestCase):
    def test_existing_dataset_never_overwritten(self):
        api = MagicMock()
        api.repo_info.return_value.sha = "1234"
        self.assertFalse(ensure_dataset(api, "owner/data"))
        api.create_repo.assert_not_called()

    def test_missing_dataset_created_and_checked(self):
        api = MagicMock()
        api.repo_info.side_effect = [RepositoryNotFoundError("404 missing"), MagicMock(sha="abc")]
        self.assertTrue(ensure_dataset(api, "owner/data", token="test"))
        api.create_repo.assert_called_once_with(
            repo_id="owner/data", repo_type="dataset",
            private=True, exist_ok=True, token="test",
        )
