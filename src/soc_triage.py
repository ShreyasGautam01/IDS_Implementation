# src/soc_triage.py
from typing import Dict, Tuple
import numpy as np
import pandas as pd


class SOCTriageEngine:
    """Routes calibrated multi-class and open-set flow predictions into Automated (A)
    or Escalated Analyst Triage (E) queues, supports validation-set cost optimization
    for tau_hi, and separates Mondrian coverage from OSR rejection and escalation rates.
    """

    def __init__(
        self, inverse_category_map: Dict[int, str], tau_hi: float = 0.25
    ):
        self.inverse_category_map = inverse_category_map
        self.tau_hi = tau_hi

    def tune_threshold_on_validation(
        self,
        probs_val: np.ndarray,
        pred_set_mask_val: np.ndarray,
        evidence_margin_val: np.ndarray,
        is_osr_rejected_val: np.ndarray,
        y_val: np.ndarray,
        cost_fn: float = 10.0,
        cost_fp: float = 2.0,
        cost_escalate: float = 1.0,
    ) -> float:
        set_sizes = np.sum(pred_set_mask_val, axis=1)
        top_closed = np.argmax(probs_val, axis=1)
        final_pred = np.where(is_osr_rejected_val, -1, top_closed)

        best_tau = self.tau_hi
        best_cost = float("inf")

        for cand_tau in np.linspace(0.05, 0.50, 19):
            is_auto = (
                (set_sizes == 1)
                & (~is_osr_rejected_val)
                & (evidence_margin_val >= cand_tau)
            )
            auto_errors = is_auto & (final_pred != y_val)
            fn_auto = np.sum(auto_errors & (y_val > 0))
            fp_auto = np.sum(auto_errors & (y_val == 0))
            n_esc = np.sum(~is_auto)

            total_cost = (
                cost_fn * fn_auto + cost_fp * fp_auto + cost_escalate * n_esc
            )
            if total_cost < best_cost:
                best_cost = total_cost
                best_tau = float(np.round(cand_tau, 4))

        self.tau_hi = best_tau
        return self.tau_hi

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

        final_pred = np.where(is_zero_day_rejected, -1, top_closed_class)

        is_automated = (
            (set_sizes == 1)
            & (~is_zero_day_rejected)
            & (evidence_margin >= self.tau_hi)
        )

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

        category_rows = []
        for cat_id in sorted(list(set(y_true))):
            mask = y_true == cat_id
            cat_name = self.inverse_category_map.get(cat_id, "Zero-Day_Unknown")
            count = int(np.sum(mask))
            recall = float(np.mean(final_pred[mask] == cat_id))
            osr_rej_rate = float(np.mean(is_zero_day_rejected[mask]))

            if cat_id >= 0:
                mondrian_cov = round(
                    float(np.mean(pred_set_mask[mask, cat_id])) * 100, 2
                )
            else:
                mondrian_cov = np.nan

            auto_mask = mask & is_automated
            auto_count = int(np.sum(auto_mask))
            esc_count = count - auto_count
            auto_rate = float(np.mean(is_automated[mask]))
            esc_rate = 1.0 - auto_rate
            auto_acc = (
                round(float(np.mean(final_pred[auto_mask] == cat_id)) * 100, 2)
                if auto_count > 0
                else np.nan
            )

            category_rows.append({
                "Category": cat_name,
                "Flows": count,
                "Recall": round(recall * 100, 2),
                "Coverage": mondrian_cov,
                "OSR_Reject_Pct": round(osr_rej_rate * 100, 2),
                "Auto_Share": round(auto_rate * 100, 2),
                "Escalated_Share": round(esc_rate * 100, 2),
                "Auto_Flows": auto_count,
                "Escalated_Flows": esc_count,
                "Auto_Acc": auto_acc,
            })

        category_df = pd.DataFrame(category_rows)

        order_margin = np.argsort(-evidence_margin)
        errors_margin = (final_pred[order_margin] != y_true[order_margin]).astype(float)
        aurc_margin = float(
            np.mean(np.cumsum(errors_margin) / np.arange(1, n_samples + 1))
        )

        gate_eligible = ((set_sizes == 1) & (~is_zero_day_rejected)).astype(float)
        composite_priority = gate_eligible * 10.0 + evidence_margin
        order_gate = np.argsort(-composite_priority)
        errors_gate = (final_pred[order_gate] != y_true[order_gate]).astype(float)
        aurc_gate = float(
            np.mean(np.cumsum(errors_gate) / np.arange(1, n_samples + 1))
        )

        benign_mask = y_true == 0
        far = (
            float(np.mean(final_pred[benign_mask] != 0))
            if np.any(benign_mask)
            else 0.0
        )
        auto_sub_acc = (
            float(np.mean(final_pred[is_automated] == y_true[is_automated])) * 100
            if np.any(is_automated)
            else np.nan
        )

        atk_auto_mask = is_automated & (y_true != 0)
        atk_auto_acc = (
            float(np.mean(final_pred[atk_auto_mask] == y_true[atk_auto_mask])) * 100
            if np.any(atk_auto_mask)
            else np.nan
        )

        global_metrics = {
            "Tuned_Tau_Hi": round(self.tau_hi, 4),
            "Automated_Coverage_Rate": round(float(np.mean(is_automated)) * 100, 2),
            "Automated_Subset_Accuracy": round(auto_sub_acc, 2),
            "Automated_Attack_Only_Accuracy": round(atk_auto_acc, 2),
            "False_Alarm_Rate_FAR": round(far * 100, 2),
            "Triage_AURC": round(aurc_margin, 4),
            "Gate_Consistent_AURC": round(aurc_gate, 4),
        }
        return flow_records, category_df, global_metrics
