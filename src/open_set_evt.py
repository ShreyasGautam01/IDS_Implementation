# src/open_set_evt.py
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.stats import weibull_min
from sklearn.decomposition import PCA


class DualStageOpenSetDetector:
    """Stage 1: Covariance-aware Mahalanobis (or native Euclidean) distance fitted on Z_t
    with Weibull EVT tails calibrated on held-out Z_c and probability-conserving attenuation.
    Stage 2: Category-specific PCA subspace reconstruction verification (Q_0.98 on Z_c)
    and validation-calibrated rejection threshold on Z_v.
    """

    def __init__(
        self,
        distance_metric: str = "mahalanobis",
        tail_size_eta: int = 50,
        attenuation_vartheta: float = 0.85,
        rec_quantile_theta: float = 0.98,
        osr_fpr_target: float = 0.03,
    ):
        assert distance_metric in ("mahalanobis", "euclidean")
        self.distance_metric = distance_metric
        self.tail_size_eta = tail_size_eta
        self.attenuation_vartheta = attenuation_vartheta
        self.rec_quantile_theta = rec_quantile_theta
        self.osr_fpr_target = osr_fpr_target

        self.mavs_: Dict[int, np.ndarray] = {}
        self.inv_covs_: Dict[int, np.ndarray] = {}
        self.weibull_params_: Dict[int, Tuple[float, float, float]] = {}
        self.class_reconstructors_: Dict[int, PCA] = {}
        self.rec_thresholds_: Dict[int, float] = {}
        self.osr_threshold_: float = 0.50
        self.known_classes_: np.ndarray = np.array([])
        self.cal_fallback_classes_: int = 0

    def _compute_distances(self, V: np.ndarray, c: int) -> np.ndarray:
        diff = V - self.mavs_[c]
        if self.distance_metric == "euclidean":
            return np.sqrt(np.maximum(np.sum(diff * diff, axis=1), 1e-12))
        left = diff @ self.inv_covs_[c]
        return np.sqrt(np.maximum(np.sum(left * diff, axis=1), 1e-12))

    def _extract_probs(self, V: np.ndarray, logits: np.ndarray) -> np.ndarray:
        k_classes = len(self.known_classes_)
        raw_m = logits if logits.shape[1] == k_classes else V[:, -k_classes:]
        shifted = raw_m - np.max(raw_m, axis=1, keepdims=True)
        exp_m = np.exp(shifted)
        return exp_m / np.sum(exp_m, axis=1, keepdims=True)

    def _compute_raw_osr_score(
        self, V: np.ndarray, logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = V.shape[0]
        probs = self._extract_probs(V, logits)
        attenuated_probs = probs.copy()

        for idx, c in enumerate(self.known_classes_):
            d_c = self._compute_distances(V, c)
            shape_k, loc, scale_lam = self.weibull_params_[c]
            w_cdf = weibull_min.cdf(d_c, shape_k, loc=loc, scale=scale_lam)
            attenuated_probs[:, idx] = probs[:, idx] * (
                1.0 - self.attenuation_vartheta * w_cdf
            )

        transferred_mass = np.sum(probs - attenuated_probs, axis=1)
        top_known_idx = np.argmax(probs, axis=1)

        rec_ratio = np.zeros(n_samples, dtype=np.float64)
        for idx, c in enumerate(self.known_classes_):
            sample_mask = top_known_idx == idx
            if np.any(sample_mask):
                pca_c = self.class_reconstructors_[c]
                V_sub = V[sample_mask]
                V_rec = pca_c.inverse_transform(pca_c.transform(V_sub))
                errors = np.mean((V_sub - V_rec) ** 2, axis=1)
                rec_ratio[sample_mask] = errors / max(
                    self.rec_thresholds_[c], 1e-12
                )

        base_uncertainty = 1.0 - np.max(probs, axis=1)
        subspace_excess = 1.0 - np.exp(-np.maximum(0.0, rec_ratio - 1.0))
        zero_day_prob = np.clip(
            0.40 * base_uncertainty
            + 0.40 * transferred_mass
            + 0.20 * subspace_excess,
            0.0,
            1.0,
        )
        return zero_day_prob, rec_ratio > 1.0

    def fit(
        self,
        V_train: np.ndarray,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        V_cal: np.ndarray,
        y_cal: np.ndarray,
        V_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "DualStageOpenSetDetector":
        self.known_classes_ = np.unique(y_train)
        dim = V_train.shape[1]
        self.cal_fallback_classes_ = 0

        for c in self.known_classes_:
            mask_correct = (y_train == c) & (y_pred_train == c)
            if np.sum(mask_correct) < 10:
                mask_correct = y_train == c
            V_c = V_train[mask_correct]

            self.mavs_[c] = np.mean(V_c, axis=0)
            if self.distance_metric == "euclidean" or V_c.shape[0] <= 1:
                self.inv_covs_[c] = np.eye(dim)
            else:
                cov_c = np.cov(V_c, rowvar=False) + 1e-3 * np.eye(dim)
                self.inv_covs_[c] = np.linalg.pinv(cov_c)

            n_comp = min(16, max(1, V_c.shape[0] - 1), max(1, V_c.shape[1] - 1))
            pca_c = PCA(n_components=n_comp, random_state=42).fit(V_c)
            self.class_reconstructors_[c] = pca_c

            # Calibrate Weibull EVT tail and PCA threshold on held-out Z_c to prevent in-sample margin overfitting
            if np.sum(y_cal == c) >= 10:
                V_cal_c = V_cal[y_cal == c]
            else:
                self.cal_fallback_classes_ += 1
                V_cal_c = V_c

            d_cal_c = self._compute_distances(V_cal_c, c)
            sorted_d = np.sort(d_cal_c)
            tail_d = np.maximum(
                sorted_d[-min(len(sorted_d), self.tail_size_eta) :], 1e-6
            )
            if np.ptp(tail_d) < 1e-6:
                tail_d = tail_d + np.linspace(1e-6, 1e-3, len(tail_d))
            shape_k, loc, scale_lam = weibull_min.fit(tail_d, floc=0)
            self.weibull_params_[c] = (shape_k, loc, scale_lam)

            V_rec_cal = pca_c.inverse_transform(pca_c.transform(V_cal_c))
            rec_errors = np.mean((V_cal_c - V_rec_cal) ** 2, axis=1)
            self.rec_thresholds_[c] = float(
                np.quantile(rec_errors, self.rec_quantile_theta)
            )

        # Calibrate OSR rejection threshold on Z_v to bound false rejection on known traffic <= osr_fpr_target
        ref_V = V_val if V_val is not None else V_cal
        ref_scores, _ = self._compute_raw_osr_score(ref_V, ref_V)
        self.osr_threshold_ = float(
            np.quantile(ref_scores, 1.0 - self.osr_fpr_target)
        )
        return self

    def score_open_set(
        self, V: np.ndarray, logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        zero_day_prob, rec_reject = self._compute_raw_osr_score(V, logits)
        is_zero_day_rejected = (zero_day_prob > self.osr_threshold_) | rec_reject
        return zero_day_prob, is_zero_day_rejected
