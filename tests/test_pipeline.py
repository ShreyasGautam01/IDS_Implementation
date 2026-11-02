# tests/test_pipeline.py
import unittest
import numpy as np
import pandas as pd

from src.conformal_mondrian import MondrianConformalCalibrator
from src.data_forensics import ForensicDataSanitizer
from src.models import HybridNIDSBackbone
from src.open_set_evt import DualStageOpenSetDetector
from src.partitioning import create_temporal_four_way_split
from src.soc_triage import SOCTriageEngine


class TestNIDSPipelineIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        n_rows = 600
        df = pd.DataFrame({
            "L4_SRC_PORT": rng.integers(1024, 65535, size=n_rows),
            "L4_DST_PORT": rng.integers(80, 8080, size=n_rows),
            "MIN_TTL": rng.integers(32, 128, size=n_rows),
            "TCP_WIN_MAX_IN": rng.integers(1024, 65535, size=n_rows),
            "PROTOCOL": rng.integers(1, 17, size=n_rows),
            "IN_BYTES": rng.exponential(500, size=n_rows),
            "IN_PKTS": rng.integers(1, 50, size=n_rows),
            "OUT_BYTES": rng.exponential(500, size=n_rows),
            "OUT_PKTS": rng.integers(1, 50, size=n_rows),
            "FLOW_DURATION_MILLISECONDS": rng.exponential(100, size=n_rows),
            "Attack": rng.choice(
                ["Benign", "Exploits", "Fuzzers", "Worms"],
                size=n_rows,
                p=[0.60, 0.20, 0.15, 0.05],
            ),
        })
        df["Label"] = (df["Attack"] != "Benign").astype(int)
        # Inject exact duplicates at start (guaranteed in Z_t)
        self.df_raw = pd.concat(
            [df.iloc[:25], df.iloc[:25], df]
        ).reset_index(drop=True)

    def test_end_to_end_protocol_guarantees(self) -> None:
        sanitizer = ForensicDataSanitizer(ablation_level="Set_C")
        df_clean = sanitizer.clean_and_harmonize(self.df_raw, is_train_reference=True)
        self.assertEqual(len(df_clean), len(self.df_raw))
        self.assertEqual(len(sanitizer.feature_columns_), 5)

        X, y = sanitizer.encode_categories(df_clean, zero_day_families=["Worms"])

        split, _ = create_temporal_four_way_split(X, y, deduplicate_train_only=True)
        self.assertGreaterEqual(split.metadata["Train_Duplicates_Removed"], 20)
        self.assertEqual(
            split.metadata["Test_Total_Rows"],
            split.metadata["Test_Known_Rows"] + split.metadata["Test_ZeroDay_Rows"],
        )

        backbone = HybridNIDSBackbone(
            feature_names=sanitizer.feature_columns_, random_state=42
        ).fit(split.X_train, split.y_train)

        X_probe = split.X_test[:25].copy()
        X_padded = X_probe.copy()
        X_padded[:, :4] += 3.0
        p_orig = backbone.predict_proba(X_probe)
        p_pad = backbone.predict_proba(X_padded)
        self.assertLessEqual(float(np.max(p_pad[:, 0] - p_orig[:, 0])), 1e-7)

        V_tr, _ = backbone.extract_penultimate_embeddings(split.X_train)
        V_va, _ = backbone.extract_penultimate_embeddings(split.X_val)
        V_ca, _ = backbone.extract_penultimate_embeddings(split.X_cal)
        V_te, logits_te = backbone.extract_penultimate_embeddings(split.X_test)
        y_pred_tr = np.argmax(backbone.predict_proba(split.X_train), axis=1)

        osr = DualStageOpenSetDetector(distance_metric="mahalanobis").fit(
            V_tr, split.y_train, y_pred_tr, V_ca, split.y_cal, V_va, split.y_val
        )
        zd_prob, is_zd_rej = osr.score_open_set(V_te, logits_te)
        self.assertEqual(len(zd_prob), len(split.y_test))

        probs_ca = backbone.predict_proba(split.X_cal)
        probs_te = backbone.predict_proba(split.X_test)
        mcp = MondrianConformalCalibrator(alpha=0.10, random_state=42).fit(
            probs_ca, split.y_cal
        )
        pred_sets, margins, pvals = mcp.predict_sets_and_margins(probs_te)
        self.assertTrue(np.all((pvals >= 0.0) & (pvals <= 1.0)))

        triage = SOCTriageEngine(sanitizer.inverse_category_map_, tau_hi=0.05)
        _, cat_df, global_metrics = triage.route_and_evaluate(
            probs_te, pred_sets, margins, zd_prob, is_zd_rej, split.y_test
        )
        zd_row = cat_df[cat_df["Category"] == "Zero-Day_Unknown"].iloc[0]
        self.assertTrue(np.isnan(zd_row["Coverage"]))
        self.assertIn("Gate_Consistent_AURC", global_metrics)


if __name__ == "__main__":
    unittest.main()
