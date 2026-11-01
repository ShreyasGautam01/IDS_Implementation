# src/soc_triage.py
from typing import Dict, Tuple
import numpy as np
import pandas as pd


class SOCTriageEngine:
    """Routes calibrated multi-class and open-set flow predictions into Automated (A)
    or Escalated Analyst Triage (E) queues and generates per-category telemetry.
    """

    def __init__(
        self, inverse_category_map: Dict[int, str], tau_hi: float = 0.25
    ):
        self.inverse_category_map = inverse_category_map
        self.tau_hi = tau_hi

    def route_and_evaluate(
        self,
        probs_test: np.ndarray,
        pred_set_mask: np.ndarray,
        evidence_margin: np.ndarray,
        zero_day_prob: np.ndarray,
        is_zero_day_rejected: np.ndarray,
        y_true: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        n_samples = len(y_true)
        set_sizes = np.sum(pred_set_mask, axis=1)
        top_closed_class = np.argmax(probs_test, axis=1)

        # Final predicted category overrides to -1 (Zero-Day_Unknown) when Open-Set rejection triggers
        final_pred = np.where(is_zero_day_rejected, -1, top_closed_class)

        # Selective Classification Routing Gate: Automated (A) vs Escalated (E)
        is_automated = (
            (set_sizes == 1)
            & (~is_zero_day_rejected)
            & (evidence_margin >= self.tau_hi)
        )

        # Build flow-level SOC output table
        candidate_sets_str = []
        for i in range(n_samples):
            if is_zero_day_rejected[i]:
                candidate_sets_str.append("{Zero-Day_Unknown}")
            else:
                members = [
                    self.inverse_category_map[c]
                    for c in np.where(pred_set_mask[i])[0]
                ]
                candidate_sets_str.append(
                    "{" + ", ".join(members) + "}" if members else "{Empty_Reject}"
                )

        flow_records = pd.DataFrame({
            "True_Category": [
                self.inverse_category_map.get(y, "Zero-Day_Unknown")
                for y in y_true
            ],
            "Predicted_Category": [
                self.inverse_category_map.get(y, "Zero-Day_Unknown")
                for y in final_pred
            ],
            "Conformal_Prediction_Set": candidate_sets_str,
            "Set_Size": set_sizes,
            "Evidence_Margin": np.round(evidence_margin, 4),
            "Zero_Day_Prob": np.round(zero_day_prob, 4),
            "Triage_Queue": np.where(
                is_automated, "Automated (A)", "Escalated (E)"
            ),
        })

        # Compute Per-Category Flow Performance Summary
        category_rows = []
        for cat_id in sorted(list(set(y_true))):
            mask = y_true == cat_id
            cat_name = self.inverse_category_map.get(cat_id, "Zero-Day_Unknown")
            count = int(np.sum(mask))
            recall = float(np.mean(final_pred[mask] == cat_id))

            if cat_id >= 0:
                empirical_cov = float(np.mean(pred_set_mask[mask, cat_id]))
            else:
                empirical_cov = float(np.mean(is_zero_day_rejected[mask]))

            auto_mask = mask & is_automated
            auto_rate = float(np.mean(is_automated[mask]))
            auto_acc = (
                float(np.mean(final_pred[auto_mask] == cat_id))
                if np.any(auto_mask)
                else np.nan
            )

            category_rows.append({
                "Category": cat_name,
                "Flows": count,
                "Recall": round(recall * 100, 2),
                "Coverage": round(empirical_cov * 100, 2),
                "Auto_Share": round(auto_rate * 100, 2),
                "Auto_Acc": (
                    round(auto_acc * 100, 2)
                    if not np.isnan(auto_acc)
                    else 100.0
                ),
            })

        category_df = pd.DataFrame(category_rows)

        # Compute Area Under the Risk-Coverage Curve (AURC)
        order = np.argsort(-evidence_margin)
        sorted_errors = (final_pred[order] != y_true[order]).astype(float)
        cumulative_risk = np.cumsum(sorted_errors) / np.arange(
            1, n_samples + 1
        )
        aurc = float(np.mean(cumulative_risk))

        # Compute False Alarm Rate (FAR) on Benign traffic (Category 0)
        benign_mask = y_true == 0
        far = (
            float(np.mean(final_pred[benign_mask] != 0))
            if np.any(benign_mask)
            else 0.0
        )

        # Safe accuracy computation on the Automated (A) subset
        auto_sub_acc = (
            float(np.mean(final_pred[is_automated] == y_true[is_automated])) * 100
            if np.any(is_automated)
            else 0.0
        )

        global_metrics = {
            "Automated_Coverage_Rate": round(float(np.mean(is_automated)) * 100, 2),
            "Automated_Subset_Accuracy": round(auto_sub_acc, 2),
            "False_Alarm_Rate_FAR": round(far * 100, 2),
            "Triage_AURC": round(aurc, 4),
        }
        return flow_records, category_df, global_metrics