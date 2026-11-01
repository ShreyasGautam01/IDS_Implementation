# src/models.py
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


class HybridNIDSBackbone:
    """Combines Monotone Gradient-Boosted Trees (for calibrated closed-set probabilities)
    with a Penultimate Feature Encoder (for Mahalanobis EVT Open-Set Recognition).
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

        # Enforce positive monotonicity on volumetric features for binary classification
        self.monotonic_cst = [
            1
            if any(k in col.upper() for k in volumetric_keywords)
            else 0
            for col in feature_names
        ]

        self.tree_backbone = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=63,
            monotonic_cst=self.monotonic_cst,
            class_weight="balanced",
            early_stopping=True,
            random_state=random_state,
        )

        # Neural encoder for penultimate latent space representations v(x)
        self.latent_encoder = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=100,
            early_stopping=True,
            random_state=random_state,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HybridNIDSBackbone":
        # scikit-learn supports monotonic_cst on binary targets; disable for multi-class
        if len(np.unique(y_train)) > 2:
            self.tree_backbone.set_params(monotonic_cst=None)
        else:
            self.tree_backbone.set_params(monotonic_cst=self.monotonic_cst)

        self.tree_backbone.fit(X_train, y_train)
        self.latent_encoder.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns closed-set probability distribution over known classes {0..K-1}."""
        return self.tree_backbone.predict_proba(X)

    def extract_penultimate_embeddings(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through MLP layers to extract penultimate 64-D activation vector v(x) and raw logits."""
        a1 = np.maximum(
            0, X @ self.latent_encoder.coefs_[0] + self.latent_encoder.intercepts_[0]
        )
        v_x = np.maximum(
            0,
            a1 @ self.latent_encoder.coefs_[1] + self.latent_encoder.intercepts_[1],
        )
        logits = (
            v_x @ self.latent_encoder.coefs_[2] + self.latent_encoder.intercepts_[2]
        )
        return v_x, logits