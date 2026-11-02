# src/conformal_mondrian.py
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import ks_2samp


class MondrianConformalCalibrator:
    """Inductive Class-Conditional (Mondrian) Conformal Predictor with explicit
    tracking of calibration stratum sizes and KS fallback activations.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        allow_ks_fallback: bool = True,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.allow_ks_fallback = allow_ks_fallback
        self.min_stratum_size = int(np.ceil((1.0 - alpha) / alpha))
        self.rng = np.random.default_rng(random_state)
        self.cal_scores_by_class_: Dict[int, np.ndarray] = {}
        self.stratum_counts_: Dict[int, int] = {}
        self.ks_fallback_activated_classes_: List[int] = []
        self.classes_: np.ndarray = np.array([])

    def fit(
        self, probs_cal: np.ndarray, y_cal: np.ndarray
    ) -> "MondrianConformalCalibrator":
        self.classes_ = np.arange(probs_cal.shape[1])
        raw_strata: Dict[int, np.ndarray] = {}
        self.ks_fallback_activated_classes_ = []

        for c in self.classes_:
            mask_c = y_cal == c
            self.stratum_counts_[int(c)] = int(np.sum(mask_c))
            if np.any(mask_c):
                raw_strata[c] = 1.0 - probs_cal[mask_c, c]
            else:
                raw_strata[c] = np.array([], dtype=np.float64)

        well_sampled = [
            c
            for c, s in raw_strata.items()
            if len(s) >= self.min_stratum_size
        ]

        for c in self.classes_:
            scores_c = raw_strata[c]
            if (
                self.allow_ks_fallback
                and len(scores_c) < self.min_stratum_size
                and len(well_sampled) > 0
            ):
                self.ks_fallback_activated_classes_.append(int(c))
                if len(scores_c) >= 2:
                    best_match = min(
                        well_sampled,
                        key=lambda ref_c: ks_2samp(
                            scores_c, raw_strata[ref_c]
                        ).statistic,
                    )
                else:
                    attack_well = [w for w in well_sampled if w > 0] or well_sampled
                    best_match = max(
                        attack_well, key=lambda ref_c: len(raw_strata[ref_c])
                    )
                pooled_scores = np.concatenate([scores_c, raw_strata[best_match]])
                self.cal_scores_by_class_[c] = np.sort(pooled_scores)
            elif len(scores_c) == 0:
                self.cal_scores_by_class_[c] = np.array([1.0], dtype=np.float64)
            else:
                self.cal_scores_by_class_[c] = np.sort(scores_c)

        return self

    def compute_mondrian_p_values(self, probs_test: np.ndarray) -> np.ndarray:
        n_test, n_classes = probs_test.shape
        p_values = np.zeros((n_test, n_classes), dtype=np.float64)
        theta = self.rng.uniform(0.0, 1.0, size=n_test)
        tol = 1e-7

        for c in self.classes_:
            cal_s = self.cal_scores_by_class_[c]
            n_c = len(cal_s)
            test_s = 1.0 - probs_test[:, c]

            idx_left = np.searchsorted(cal_s, test_s - tol, side="left")
            idx_right = np.searchsorted(cal_s, test_s + tol, side="right")

            greater_count = n_c - idx_right
            equal_count = idx_right - idx_left

            p_values[:, c] = (
                greater_count + theta * (equal_count + 1.0)
            ) / (n_c + 1.0)

        return p_values

    def predict_sets_and_margins(
        self, probs_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_values = self.compute_mondrian_p_values(probs_test)
        pred_set_mask = p_values > self.alpha

        sorted_p = np.sort(p_values, axis=1)
        evidence_margin = sorted_p[:, -1] - sorted_p[:, -2]
        return pred_set_mask, evidence_margin, p_values
