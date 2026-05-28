"""Load and cache the Bitext Customer Service dataset as a pandas DataFrame.

The CSV is gitignored; if it is missing (e.g. a fresh clone) it is downloaded
from Hugging Face once and cached under ``data/``. Subsequent calls reuse the
in-process cache.
"""

import shutil
from functools import lru_cache
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH: Path = _DATA_DIR / "bitext_customer_support.csv"
EXPECTED_COLUMNS: tuple[str, ...] = ("flags", "instruction", "category", "intent", "response")

_HF_REPO: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
_HF_FILE: str = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"


def _ensure_dataset() -> Path:
    """Return the local CSV path, downloading it from Hugging Face if absent."""
    if not DATASET_PATH.exists():
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(_HF_REPO, _HF_FILE, repo_type="dataset")
        shutil.copyfile(downloaded, DATASET_PATH)
    return DATASET_PATH


@lru_cache(maxsize=1)
def load_dataframe() -> pd.DataFrame:
    """Return the dataset as a DataFrame, loaded once and cached per process."""
    return pd.read_csv(_ensure_dataset())
