# src/models.py
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class HybridNIDSBackbone:
    """Couples a Binary Monotone GBDT (enforcing positive volumetric monotonicity)
    with a Multiclass GBDT so volumetric monotonicity holds across both binary
    and multi-class intrusion detection, and extracts joint feature-margin embeddings.
    """

    def __init__(
        self,
        feature_names: List[str],
        volumetric_keywords: Tuple[str, ...] = (
            "BYTES",
            "PKTS",
            "THROUGHPUT",
        ),
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.random_state = random_state
        self.is_multiclass_: bool = False

        self.monotonic_cst = [
            1
            if any(k in col.upper() for k in volumetric_keywords)
            else 0
            for col in feature_names
        ]

        self.binary_monotone_gbdt = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=63,
            monotonic_cst=self.monotonic_cst,
            class_weight="balanced",
            early_stopping=True,
            random_state=random_state,
        )

        self.tree_backbone = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=63,
            monotonic_cst=None,
            class_weight="balanced",
            early_stopping=True,
            random_state=random_state,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HybridNIDSBackbone":
        unique_classes = np.unique(y_train)
        self.is_multiclass_ = len(unique_classes) > 2

        if self.is_multiclass_:
            y_bin = (y_train > 0).astype(np.int64)
            self.binary_monotone_gbdt.fit(X_train, y_bin)
            self.tree_backbone.fit(X_train, y_train)
        else:
            self.tree_backbone.set_params(monotonic_cst=self.monotonic_cst)
            self.tree_backbone.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns monotone-coupled probability distribution over {0..K-1}."""
        if not self.is_multiclass_:
            return self.tree_backbone.predict_proba(X)

        p_multi = self.tree_backbone.predict_proba(X)
        p_atk_mono = self.binary_monotone_gbdt.predict_proba(X)[:, 1:2]
        p_benign_mono = 1.0 - p_atk_mono

        atk_mass = np.sum(p_multi[:, 1:], axis=1, keepdims=True)
        n_atk_classes = p_multi.shape[1] - 1
        atk_cond = np.where(
            atk_mass > 1e-12, p_multi[:, 1:] / atk_mass, 1.0 / n_atk_classes
        )
        return np.hstack([p_benign_mono, p_atk_mono * atk_cond])

    def extract_penultimate_embeddings(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenates standardized flow features X with raw GBDT decision margins g(X)
        for covariance/PCA modeling, and returns raw margins as the second element.
        """
        raw_margins = self.tree_backbone.decision_function(X)
        if raw_margins.ndim == 1:
            raw_margins = np.column_stack([-raw_margins, raw_margins])
        V = np.hstack([X, raw_margins])
        return V, raw_margins
