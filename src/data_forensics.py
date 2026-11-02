# src/data_forensics.py
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

EXPLICIT_IDENTIFIERS = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "FLOW_ID",
    "TIMESTAMP",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "Dataset",
]

LEAKAGE_GROUPS = {
    "Set_A_TTL": ["MIN_TTL", "MAX_TTL", "STTL", "DTTL"],
    "Set_B_Window": [
        "TCP_WIN_MAX_IN",
        "TCP_WIN_MAX_OUT",
        "INIT_WIN_BYTES_FWD",
        "INIT_WIN_BYTES_BWD",
    ],
    "Set_C_Protocol_IDs": [
        "PROTOCOL",
        "L7_PROTO",
        "ICMP_TYPE",
        "ICMP_IPV4_TYPE",
        "DNS_QUERY_ID",
    ],
}


class ForensicDataSanitizer:
    """Sanitizes flow records, logs row-level filter counts, and locks feature schemas
    without altering evaluation splits via pre-split deduplication.
    """

    def __init__(
        self,
        ablation_level: str = "Set_C",
        label_col: str = "Attack",
        binary_col: str = "Label",
    ):
        self.ablation_level = ablation_level
        self.label_col = label_col
        self.binary_col = binary_col
        self.feature_columns_: List[str] = []
        self.dropped_columns_: List[str] = []
        self.category_map_: Dict[str, int] = {}
        self.inverse_category_map_: Dict[int, str] = {}
        self.audit_log_: Dict[str, int] = {}

    def _get_columns_to_drop(self, df_cols: List[str]) -> List[str]:
        drop_set = set(EXPLICIT_IDENTIFIERS)
        if self.ablation_level in ["Set_A", "Set_B", "Set_C"]:
            drop_set.update(LEAKAGE_GROUPS["Set_A_TTL"])
        if self.ablation_level in ["Set_B", "Set_C"]:
            drop_set.update(LEAKAGE_GROUPS["Set_B_Window"])
        if self.ablation_level == "Set_C":
            drop_set.update(LEAKAGE_GROUPS["Set_C_Protocol_IDs"])
        return [
            c
            for c in df_cols
            if c.upper() in {x.upper() for x in drop_set}
            or c in [self.label_col, self.binary_col]
        ]

    def clean_and_harmonize(
        self, df: pd.DataFrame, is_train_reference: bool = True
    ) -> pd.DataFrame:
        df = df.copy()
        n_raw = len(df)
        df.columns = [c.strip() for c in df.columns]
        if self.label_col in df.columns:
            df[self.label_col] = (
                df[self.label_col].astype(str).str.strip().replace({"": np.nan})
            )

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[self.label_col])
        n_after_label = len(df)

        duration_cols = [c for c in df.columns if "DURATION" in c.upper()]
        for d_col in duration_cols:
            df = df[df[d_col] >= 0]
        n_after_dur = len(df)

        pkt_cols = [c for c in ["IN_PKTS", "OUT_PKTS"] if c in df.columns]
        if len(pkt_cols) == 2:
            df = df[(df[pkt_cols[0]] + df[pkt_cols[1]]) > 0]
        n_after_pkts = len(df)

        if is_train_reference:
            drop_cols = self._get_columns_to_drop(list(df.columns))
            self.feature_columns_ = [
                c
                for c in df.columns
                if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
            self.dropped_columns_ = [
                c
                for c in df.columns
                if c not in self.feature_columns_
                and c not in [self.label_col, self.binary_col]
            ]

        df[self.feature_columns_] = (
            df[self.feature_columns_]
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-1e15, upper=1e15)
        )
        df = df.dropna(subset=self.feature_columns_)
        n_final = len(df)

        self.audit_log_ = {
            "Raw_Rows": n_raw,
            "Dropped_Missing_Label": n_raw - n_after_label,
            "Dropped_Negative_Duration": n_after_label - n_after_dur,
            "Dropped_Zero_Packets": n_after_dur - n_after_pkts,
            "Dropped_NonFinite_Features": n_after_pkts - n_final,
            "Retained_Rows": n_final,
        }
        return df.reset_index(drop=True)

    def encode_categories(
        self, df: pd.DataFrame, zero_day_families: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_labels = df[self.label_col].values
        if not self.category_map_:
            unique_labels = sorted(list(set(raw_labels)))
            benign_names = {"Benign", "Normal", "BENIGN", "0"}
            known_idx = 1
            for lbl in unique_labels:
                if lbl in benign_names:
                    self.category_map_[lbl] = 0
                elif lbl in zero_day_families:
                    self.category_map_[lbl] = -1
                else:
                    self.category_map_[lbl] = known_idx
                    known_idx += 1
            self.inverse_category_map_ = {
                v: k for k, v in self.category_map_.items() if v >= 0
            }
            self.inverse_category_map_[-1] = "Zero-Day_Unknown"

        y_encoded = np.array(
            [self.category_map_.get(lbl, -1) for lbl in raw_labels],
            dtype=np.int64,
        )
        X_matrix = np.nan_to_num(
            df[self.feature_columns_].to_numpy(dtype=np.float64),
            nan=0.0,
            posinf=1e15,
            neginf=-1e15,
        ).astype(np.float32)
        return X_matrix, y_encoded
