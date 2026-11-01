# src/partitioning.py
from dataclasses import dataclass
from typing import Tuple
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


def create_temporal_four_way_split(
    X: np.ndarray,
    y: np.ndarray,
    ratios: Tuple[float, float, float, float] = (0.55, 0.15, 0.15, 0.15),
) -> Tuple[FourWayDatasetSplit, RobustScaler]:
    """Chronologically splits each known class into Z_t, Z_v, Z_c, and Z_test.

    Appends withheld zero-day flows (y == -1) exclusively to Z_test for Open-Set
    evaluation. Fits RobustScaler strictly on Z_t (Pitfall P3 mitigation).
    """
    assert np.isclose(sum(ratios), 1.0), "Split ratios must sum to 1.0"
    r_t, r_v, r_c, _ = ratios

    train_idx, val_idx, cal_idx, test_idx = [], [], [], []
    known_classes = [c for c in np.unique(y) if c >= 0]

    # Preserve temporal ordering per class stratum to prevent bursty campaign leakage
    for c in known_classes:
        c_indices = np.where(y == c)[0]
        n = len(c_indices)
        b1 = int(n * r_t)
        b2 = int(n * (r_t + r_v))
        b3 = int(n * (r_t + r_v + r_c))

        train_idx.extend(c_indices[:b1])
        val_idx.extend(c_indices[b1:b2])
        cal_idx.extend(c_indices[b2:b3])
        test_idx.extend(c_indices[b3:])

    # Route all unseen zero-day attack families strictly to the locked test set
    zero_day_indices = np.where(y == -1)[0]
    test_idx.extend(zero_day_indices)

    # Fit scaler strictly on Proper Training set Z_t
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])
    X_val_scaled = scaler.transform(X[val_idx])
    X_cal_scaled = scaler.transform(X[cal_idx])
    X_test_scaled = scaler.transform(X[test_idx])

    split = FourWayDatasetSplit(
        X_train=X_train_scaled,
        y_train=y[train_idx],
        X_val=X_val_scaled,
        y_val=y[val_idx],
        X_cal=X_cal_scaled,
        y_cal=y[cal_idx],
        X_test=X_test_scaled,
        y_test=y[test_idx],
    )
    return split, scaler