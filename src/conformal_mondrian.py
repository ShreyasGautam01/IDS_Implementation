# src/conformal_mondrian.py
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import ks_2samp


class MondrianConformalCalibrator:
    """Class-Conditional (Mondrian) Conformal Prediction with Kolmogorov-Smirnov

    Clustered Conformal Prediction (CCP) fallback for rare tail attack classes.
    """

    def __init__(self, alpha: float = 0.10, random_state: int = 42):
        self.alpha = alpha
        self.min_stratum_size = int(np.ceil((1.0 - alpha) / alpha))
        self.rng = np.random.default_rng(random_state)
        self.cal_scores_by_class_: Dict[int, np.ndarray] = {}
        self.q_hat_by_class_: Dict[int, float] = {}
        self.classes_: np.ndarray = np.array([])

    def fit(
        self, probs_cal: np.ndarray, y_cal: np.ndarray
    ) -> "MondrianConformalCalibrator":
        self.classes_ = np.arange(probs_cal.shape[1])
        raw_strata: Dict[int, np.ndarray] = {}

        for c in self.classes_:
            mask_c = y_cal == c
            if np.any(mask_c):
                raw_strata[c] = 1.0 - probs_cal[mask_c, c]
            else:
                raw_strata[c] = np.array([1.0])

        # Apply Clustered Conformal Prediction (CCP) fallback for strata with n_c < min_stratum_size
        well_sampled = [
            c
            for c, s in raw_strata.items()
            if len(s) >= self.min_stratum_size
        ]
        for c in self.classes_:
            scores_c = raw_strata[c]
            if len(scores_c) < self.min_stratum_size and len(well_sampled) > 0:
                # Find closest well-sampled class stratum via two-sample Kolmogorov-Smirnov distance
                best_match = min(
                    well_sampled,
                    key=lambda ref_c: ks_2samp(
                        scores_c, raw_strata[ref_c]
                    ).statistic
                    if len(scores_c) > 1
                    else 1.0,
                )
                pooled_scores = np.concatenate(
                    [scores_c, raw_strata[best_match]]
                )
                self.cal_scores_by_class_[c] = np.sort(pooled_scores)
            else:
                self.cal_scores_by_class_[c] = np.sort(scores_c)

            n_c = len(self.cal_scores_by_class_[c])
            q_level = min(
                1.0, np.ceil((n_c + 1) * (1.0 - self.alpha)) / n_c
            )
            self.q_hat_by_class_[c] = float(
                np.quantile(
                    self.cal_scores_by_class_[c], q_level, method="higher"
                )
            )

        return self

    def compute_mondrian_p_values(self, probs_test: np.ndarray) -> np.ndarray:
        """Computes smoothed class-conditional p-values p_{y_tilde}(x) in O(N log N) time

        and O(N) memory using binary search on sorted calibration strata.
        """
        n_test, n_classes = probs_test.shape
        p_values = np.zeros((n_test, n_classes), dtype=np.float64)
        theta = self.rng.uniform(0.0, 1.0, size=n_test)
        tol = 1e-7

        for c in self.classes_:
            cal_s = self.cal_scores_by_class_[c]  # Already sorted in fit()
            n_c = len(cal_s)
            test_s = 1.0 - probs_test[:, c]

            # Binary search for elements strictly greater and within tie tolerance
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
        """Returns:

        - pred_set_mask: Boolean matrix (N, K) indicating membership in
        prediction set Gamma^alpha(x)
        - evidence_margin: Pairwise conformal evidence margin delta^em(x) =
        p_(1)(x) - p_(2)(x)
        - p_values: Full Mondrian p-value matrix
        """
        p_values = self.compute_mondrian_p_values(probs_test)
        pred_set_mask = p_values > self.alpha

        sorted_p = np.sort(p_values, axis=1)
        evidence_margin = sorted_p[:, -1] - sorted_p[:, -2]
        return pred_set_mask, evidence_margin, p_values