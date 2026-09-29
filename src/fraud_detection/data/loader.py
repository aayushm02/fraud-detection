"""Dataset loader module using factory pattern."""
import os
import pandas as pd
from typing import Any, Dict
from fraud_detection.data.synthetic import generate_synthetic_banking_data
from fraud_detection.config import get_settings

class DatasetLoader:
    @staticmethod
    def load_credit_card(path: str) -> pd.DataFrame:
        """Load credit card dataset from CSV."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}. Please check data/README.md.")
        df = pd.read_csv(path)
        if "Class" in df.columns:
            df.rename(columns={"Class": "is_fraud"}, inplace=True)
        return df

    @staticmethod
    def load_synthetic_banking(n_samples: int = 100000) -> pd.DataFrame:
        """Load synthetic banking dataset."""
        return generate_synthetic_banking_data(n_samples=n_samples)

    @staticmethod
    def load(dataset_name: str, **kwargs: Any) -> pd.DataFrame:
        """Factory method to load datasets."""
        settings = get_settings()
        
        if dataset_name == "credit_card":
            path = kwargs.get("path", settings.data.credit_card_path)
            return DatasetLoader.load_credit_card(path)
        elif dataset_name in ("synthetic_banking", "banking"):
            n_samples = kwargs.get("n_samples", settings.data.synthetic_banking_size)
            return DatasetLoader.load_synthetic_banking(n_samples)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

# Backward compatibility wrappers
def load_credit_card(path: str) -> pd.DataFrame:
    return DatasetLoader.load_credit_card(path)

def load_synthetic_banking(n_samples: int) -> pd.DataFrame:
    return DatasetLoader.load_synthetic_banking(n_samples)

def load(dataset_name: str, **kwargs: Any) -> pd.DataFrame:
    return DatasetLoader.load(dataset_name, **kwargs)
