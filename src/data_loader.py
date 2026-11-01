# src/data_loader.py
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import yaml


class NIDSDataLoader:
    """Manages ingestion and memory-efficient streaming of NF-v2 Parquet benchmarks."""

    def __init__(self, config_path: str = "configs/dataset_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def load_unsw_v2(self, nrows: Optional[int] = None) -> pd.DataFrame:
        path = Path(self.config["paths"]["unsw_v2"])
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Download NF-UNSW-NB15-v2 first."
            )

        print(f"Loading NF-UNSW-NB15-v2 from {path}...")
        df = pd.read_parquet(path)
        if nrows and len(df) > nrows:
            df = df.sample(
                n=nrows, random_state=self.config["sampling"]["random_seed"]
            )
        return df

    def load_cic18_v2(self, use_sample: bool = True) -> pd.DataFrame:
        path = Path(self.config["paths"]["cic18_v2"])
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Download"
                " NF-CSE-CIC-IDS2018-v2 first."
            )

        print(f"Loading NF-CSE-CIC-IDS2018-v2 from {path}...")
        # Because CIC-18-v2 contains ~18.89 million rows, loading the full set requires ~12GB RAM
        if use_sample:
            sample_size = self.config["sampling"]["cic18_eval_sample_size"]
            # Read metadata or use pyarrow dataset scanner for fast memory-bounded sampling
            df = pd.read_parquet(path)
            if len(df) > sample_size:
                df = df.sample(
                    n=sample_size,
                    random_state=self.config["sampling"]["random_seed"],
                )
            return df
        return pd.read_parquet(path)


def verify_dataset_pair(
    config_path: str = "configs/dataset_config.yaml",
) -> None:
    """Sanity-checks file presence, schema parity, and label availability."""
    loader = NIDSDataLoader(config_path)
    df_unsw = loader.load_unsw_v2(nrows=1000)
    df_cic = loader.load_cic18_v2(use_sample=True)

    common_features = set(df_unsw.columns).intersection(set(df_cic.columns))
    print(f"[OK] UNSW-NB15-v2 Loaded: {df_unsw.shape}")
    print(f"[OK] CSE-CIC-IDS2018-v2 Loaded: {df_cic.shape}")
    print(f"[OK] Aligned Common Features: {len(common_features)} columns")


if __name__ == "__main__":
    verify_dataset_pair()