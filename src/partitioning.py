# src/partitioning.py
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
from sklearn.preprocessing import RobustScaler


@dataclass
class FourWayDatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    metadata: Dict[str, int] = field(default_factory=dict)


def create_temporal_four_way_split(
    X: np.ndarray,
    y: np.ndarray,
    ratios: Tuple[float, float, float, float] = (0.55, 0.15, 0.15, 0.15),
    deduplicate_train_only: bool = True,
) -> Tuple[FourWayDatasetSplit, RobustScaler]:
    """Partitions known classes sequentially into Z_t, Z_v, Z_c, and Z_test,
    appends withheld zero-day flows (y == -1) strictly to Z_test, and deduplicates
    ONLY the proper training set Z_t after splitting.
    """
    assert np.isclose(sum(ratios), 1.0), "Split ratios must sum to 1.0"
    r_t, r_v, r_c, _ = ratios

    known_mask = y >= 0
    known_indices = np.where(known_mask)[0]
    zero_day_indices = np.where(y == -1)[0]

    n_known = len(known_indices)
    b1 = int(n_known * r_t)
    b2 = int(n_known * (r_t + r_v))
    b3 = int(n_known * (r_t + r_v + r_c))

    train_idx = list(known_indices[:b1])
    val_idx = list(known_indices[b1:b2])
    cal_idx = list(known_indices[b2:b3])
    test_idx = list(known_indices[b3:])

    # Verify all known classes have minimum support in global windows;
    # if a rare campaign is localized in one file segment, stratify that class sequentially.
    localized_fallback_count = 0
    for c in np.unique(y[known_mask]):
        if (
            np.sum(y[train_idx] == c) < 20
            or np.sum(y[val_idx] == c) < 10
            or np.sum(y[cal_idx] == c) < 10
        ):
            localized_fallback_count += 1
            c_idx = np.where(y == c)[0]
            nc = len(c_idx)
            cb1, cb2, cb3 = (
                int(nc * r_t),
                int(nc * (r_t + r_v)),
                int(nc * (r_t + r_v + r_c)),
            )
            train_idx = [i for i in train_idx if y[i] != c] + list(c_idx[:cb1])
            val_idx = [i for i in val_idx if y[i] != c] + list(c_idx[cb1:cb2])
            cal_idx = [i for i in cal_idx if y[i] != c] + list(c_idx[cb2:cb3])
            test_idx = [i for i in test_idx if y[i] != c] + list(c_idx[cb3:])

    train_idx.sort()
    val_idx.sort()
    cal_idx.sort()
    test_idx.sort()
    test_idx.extend(zero_day_indices)

    X_tr_raw, y_tr_raw = X[train_idx], y[train_idx]
    n_tr_initial = len(y_tr_raw)

    # Deduplicate strictly on Proper Training Set Z_t (never alter Z_v, Z_c, or Z_test)
    if deduplicate_train_only:
        combined_tr = np.column_stack([X_tr_raw, y_tr_raw])
        _, unique_indices = np.unique(combined_tr, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        X_tr_raw = X_tr_raw[unique_indices]
        y_tr_raw = y_tr_raw[unique_indices]

    n_tr_dedup = len(y_tr_raw)

    scaler = RobustScaler()
    X_train_scaled = np.clip(scaler.fit_transform(X_tr_raw), -1e6, 1e6)
    X_val_scaled = np.clip(scaler.transform(X[val_idx]), -1e6, 1e6)
    X_cal_scaled = np.clip(scaler.transform(X[cal_idx]), -1e6, 1e6)
    X_test_scaled = np.clip(scaler.transform(X[test_idx]), -1e6, 1e6)

    metadata = {
        "Train_Initial_Rows": n_tr_initial,
        "Train_Duplicates_Removed": n_tr_initial - n_tr_dedup,
        "Train_Final_Rows": n_tr_dedup,
        "Val_Rows": len(val_idx),
        "Cal_Rows": len(cal_idx),
        "Test_Known_Rows": len(test_idx) - len(zero_day_indices),
        "Test_ZeroDay_Rows": len(zero_day_indices),
        "Test_Total_Rows": len(test_idx),
        "Localized_Campaign_Fallbacks": localized_fallback_count,
    }

    split = FourWayDatasetSplit(
        X_train=X_train_scaled,
        y_train=y_tr_raw,
        X_val=X_val_scaled,
        y_val=y[val_idx],
        X_cal=X_cal_scaled,
        y_cal=y[cal_idx],
        X_test=X_test_scaled,
        y_test=y[test_idx],
        metadata=metadata,
    )
    return split, scaler