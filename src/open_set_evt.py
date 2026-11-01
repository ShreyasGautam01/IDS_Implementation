# src/open_set_evt.py
from typing import Dict, Tuple
import numpy as np
from scipy.stats import weibull_min
from sklearn.decomposition import PCA


class DualStageOpenSetDetector:
    """Stage 1: Covariance-aware Mahalanobis distance + Weibull EVT tail attenuation.

    Stage 2: Category-specific subspace reconstruction verification (theta =
    0.96 quantile).
    """

    def __init__(
        self,
        tail_size_eta: int = 50,
        attenuation_vartheta: float = 0.85,
        rec_quantile_theta: float = 0.96,
    ):
        self.tail_size_eta = tail_size_eta
        self.attenuation_vartheta = attenuation_vartheta
        self.rec_quantile_theta = rec_quantile_theta

        self.mavs_: Dict[int, np.ndarray] = {}
        self.inv_covs_: Dict[int, np.ndarray] = {}
        self.weibull_params_: Dict[int, Tuple[float, float, float]] = {}
        self.class_reconstructors_: Dict[int, PCA] = {}
        self.rec_thresholds_: Dict[int, float] = {}
        self.known_classes_: np.ndarray = np.array([])

    def _mahalanobis_distances(self, V: np.ndarray, c: int) -> np.ndarray:
        diff = V - self.mavs_[c]
        left = diff @ self.inv_covs_[c]
        return np.sqrt(np.maximum(np.sum(left * diff, axis=1), 1e-12))

    def fit(
        self,
        V_train: np.ndarray,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        V_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "DualStageOpenSetDetector":
        self.known_classes_ = np.unique(y_train)
        dim = V_train.shape[1]

        for c in self.known_classes_:
            # Filter strictly correctly classified training flows for class c
            mask_correct = (y_train == c) & (y_pred_train == c)
            if np.sum(mask_correct) < 10:
                mask_correct = y_train == c
            V_c = V_train[mask_correct]

            # 1. Compute Mean Activation Vector (MAV) and regularized inverse covariance
            self.mavs_[c] = np.mean(V_c, axis=0)
            cov_c = np.cov(V_c, rowvar=False) + 1e-4 * np.eye(dim)
            self.inv_covs_[c] = np.linalg.pinv(cov_c)

            # 2. Fit Weibull EVT distribution to the eta largest Mahalanobis distances
            d_m = self._mahalanobis_distances(V_c, c)
            sorted_d = np.sort(d_m)
            tail_d = np.maximum(
                sorted_d[-min(len(sorted_d), self.tail_size_eta) :], 1e-6
            )
            if np.ptp(tail_d) < 1e-6:
                tail_d = tail_d + np.linspace(1e-6, 1e-3, len(tail_d))
            shape_k, loc, scale_lam = weibull_min.fit(tail_d, floc=0)
            self.weibull_params_[c] = (shape_k, loc, scale_lam)

            # 3. Fit per-class latent reconstructor and calibrate theta=0.96 threshold on Z_c
            n_comp = min(16, V_c.shape[0] - 1, V_c.shape[1] - 1)
            pca_c = PCA(n_components=max(1, n_comp), random_state=42).fit(V_c)
            self.class_reconstructors_[c] = pca_c

            V_cal_c = V_cal[y_cal == c] if np.any(y_cal == c) else V_c
            V_rec_cal = pca_c.inverse_transform(pca_c.transform(V_cal_c))
            rec_errors = np.mean((V_cal_c - V_rec_cal) ** 2, axis=1)
            self.rec_thresholds_[c] = float(
                np.quantile(rec_errors, self.rec_quantile_theta)
            )

        return self

    def score_open_set(
        self, V: np.ndarray, logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns:

        - zero_day_prob: Probability mass reallocated to the unknown zero-day
        class
        - is_zero_day_rejected: Boolean flag triggered if OpenMax or Stage-2
        reconstruction rejects the flow
        """
        n_samples, n_classes = logits.shape
        attenuated_logits = logits.copy()
        weibull_Disconnect = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.known_classes_):
            d_m = self._mahalanobis_distances(V, c)
            shape_k, loc, scale_lam = self.weibull_params_[c]
            w_cdf = weibull_min.cdf(d_m, shape_k, loc=loc, scale=scale_lam)
            weibull_Disconnect[:, idx] = w_cdf
            attenuated_logits[:, idx] = logits[:, idx] * (
                1.0 - self.attenuation_vartheta * w_cdf
            )

        # Reallocate lost positive activation mass to unknown zero-day class
        unknown_logit = np.sum(
            np.maximum(0, logits - attenuated_logits), axis=1, keepdims=True
        )
        openmax_logits = np.hstack([attenuated_logits, unknown_logit])
        openmax_logits -= np.max(openmax_logits, axis=1, keepdims=True)
        exp_logits = np.exp(openmax_logits)
        openmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        zero_day_prob = openmax_probs[:, -1]
        top_known_idx = np.argmax(openmax_probs[:, :-1], axis=1)
        openmax_reject = np.argmax(openmax_probs, axis=1) == n_classes

        # Stage 2: Check per-class reconstruction loss against 0.96 calibration quantile
        rec_reject = np.zeros(n_samples, dtype=bool)
        for idx, c in enumerate(self.known_classes_):
            sample_mask = top_known_idx == idx
            if np.any(sample_mask):
                pca_c = self.class_reconstructors_[c]
                V_sub = V[sample_mask]
                V_rec = pca_c.inverse_transform(pca_c.transform(V_sub))
                errors = np.mean((V_sub - V_rec) ** 2, axis=1)
                rec_reject[sample_mask] = errors > self.rec_thresholds_[c]

        is_zero_day_rejected = openmax_reject | rec_reject
        return zero_day_prob, is_zero_day_rejected