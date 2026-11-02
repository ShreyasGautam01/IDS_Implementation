# src/data_loader.py
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import yaml


class NetFlowDatasetLoader:
    """Ingests NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 Parquet datasets using
    aligned configuration keys from configs/dataset_config.yaml.
    """

    def __init__(self, config_path: str = "configs/dataset_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.unsw_path = Path(self.config["paths"]["unsw_v2"])
        self.cic18_path = Path(self.config["paths"]["cic18_v2"])
        params = self.config.get("parameters", {})
        seeds = params.get("random_seeds", [42])
        self.default_seed = seeds[0] if isinstance(seeds, list) else int(seeds)
        self.cic18_sample_size = int(params.get("cic18_sample_size", 200000))

    def load_unsw_v2(self, max_rows: int = 0) -> pd.DataFrame:
        df = pd.read_parquet(self.unsw_path)
        if max_rows > 0 and len(df) > max_rows:
            df = (
                df.sample(n=max_rows, random_state=self.default_seed)
                .sort_index()
                .reset_index(drop=True)
            )
        return df

    def load_cic18_v2(self, sample_size: int = 0) -> pd.DataFrame:
        target_n = sample_size if sample_size > 0 else self.cic18_sample_size
        df = pd.read_parquet(self.cic18_path)
        if target_n > 0 and len(df) > target_n:
            df = (
                df.sample(n=target_n, random_state=self.default_seed)
                .sort_index()
                .reset_index(drop=True)
            )
        return df

    def verify_dataset_pair(self) -> Dict[str, Tuple[int, int]]:
        df_u = self.load_unsw_v2(max_rows=10000)
        df_c = self.load_cic18_v2(sample_size=10000)
        return {"UNSW_Sample": df_u.shape, "CIC18_Sample": df_c.shape}
