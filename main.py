# main.py
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
import yaml

from src.conformal_mondrian import MondrianConformalCalibrator
from src.data_forensics import ForensicDataSanitizer
from src.models import HybridNIDSBackbone
from src.open_set_evt import DualStageOpenSetDetector
from src.partitioning import create_temporal_four_way_split
from src.soc_triage import SOCTriageEngine


def load_config(config_path: str = "configs/dataset_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_multiclass_pipeline(
    parquet_path: str,
    ablation_level: str = "Set_C",
    zero_day_family: str = "Worms",
    max_rows: int = 0,
    alpha: float = 0.10,
    tau_hi: float = 0.25,
    seed: int = 42,
) -> None:
    print("=" * 68)
    print(
        f"RUNNING UNIFIED OSR + MONDRIAN + TRIAGE PIPELINE | Ablation:"
        f" {ablation_level} | Seed: {seed}"
    )
    print("=" * 68)
    t0 = time.time()

    df_raw = pd.read_parquet(parquet_path)
    if max_rows > 0 and len(df_raw) > max_rows:
        df_attack = df_raw[df_raw["Label"] == 1]
        n_benign_target = max(10000, max_rows - len(df_attack))
        df_benign = df_raw[df_raw["Label"] == 0].sample(
            n=n_benign_target, random_state=seed
        )
        df_raw = (
            pd.concat([df_benign, df_attack])
            .sort_index()
            .reset_index(drop=True)
        )
        print(f"[Smoke Test Mode] Using subset of {len(df_raw):,} flows.")
    else:
        print(f"[Full Mode] Loaded {len(df_raw):,} total flows.")

    attack_counts = df_raw["Attack"].value_counts()
    if zero_day_family not in attack_counts.index:
        zero_day_family = attack_counts.index[-1]

    sanitizer = ForensicDataSanitizer(
        ablation_level=ablation_level, label_col="Attack", binary_col="Label"
    )
    df_clean = sanitizer.clean_and_harmonize(df_raw, is_train_reference=True)
    X_all, y_all = sanitizer.encode_categories(
        df_clean, zero_day_families=[zero_day_family]
    )
    print(
        f"[Stage 1] Retained {len(sanitizer.feature_columns_)} features."
        f" Audit: {sanitizer.audit_log_}"
    )

    split, _ = create_temporal_four_way_split(
        X_all, y_all, ratios=(0.55, 0.15, 0.15, 0.15), deduplicate_train_only=True
    )
    print(f"[Stage 2] Split Metadata: {split.metadata}")

    print("[Stage 3] Fitting Coupled Monotone GBDT Backbone on Z_t...")
    backbone = HybridNIDSBackbone(
        feature_names=sanitizer.feature_columns_, random_state=seed
    )
    backbone.fit(split.X_train, split.y_train)

    print("[Stage 4] Fitting Mahalanobis & Native Euclidean EVT OSR on Z_t/Z_c/Z_v...")
    V_train, _ = backbone.extract_penultimate_embeddings(split.X_train)
    V_val, _ = backbone.extract_penultimate_embeddings(split.X_val)
    V_cal, _ = backbone.extract_penultimate_embeddings(split.X_cal)
    V_test, logits_test = backbone.extract_penultimate_embeddings(split.X_test)

    y_pred_train = np.argmax(backbone.predict_proba(split.X_train), axis=1)

    osr_detector = DualStageOpenSetDetector(
        distance_metric="mahalanobis",
        tail_size_eta=50,
        attenuation_vartheta=0.85,
        rec_quantile_theta=0.96,
    )
    osr_detector.fit(
        V_train, split.y_train, y_pred_train, V_cal, split.y_cal, V_val, split.y_val
    )
    zero_day_prob, is_zero_day_rejected = osr_detector.score_open_set(
        V_test, logits_test
    )
    _, is_osr_rej_val = osr_detector.score_open_set(V_val, V_val)

    print(f"[Stage 5] Calibrating Mondrian Conformal Quantiles on Z_c (alpha = {alpha})...")
    probs_val = backbone.predict_proba(split.X_val)
    probs_cal = backbone.predict_proba(split.X_cal)
    probs_test = backbone.predict_proba(split.X_test)

    mondrian_cp = MondrianConformalCalibrator(alpha=alpha, random_state=seed)
    mondrian_cp.fit(probs_cal, split.y_cal)
    print(
        f"  -> KS Fallback Activated Classes: {len(mondrian_cp.ks_fallback_activated_classes_)}"
        f" | Stratum Counts: {mondrian_cp.stratum_counts_}"
    )

    pred_set_val, margin_val, _ = mondrian_cp.predict_sets_and_margins(probs_val)
    pred_set_mask, evidence_margin, p_values_test = (
        mondrian_cp.predict_sets_and_margins(probs_test)
    )

    triage = SOCTriageEngine(
        inverse_category_map=sanitizer.inverse_category_map_, tau_hi=tau_hi
    )
    tuned_tau = triage.tune_threshold_on_validation(
        probs_val, pred_set_val, margin_val, is_osr_rej_val, split.y_val
    )
    print(f"[Stage 6] Validation-Tuned tau_hi on Z_v = {tuned_tau:.4f}")

    flow_df, category_summary, global_summary = triage.route_and_evaluate(
        probs_test,
        pred_set_mask,
        evidence_margin,
        zero_day_prob,
        is_zero_day_rejected,
        split.y_test,
    )

    print("\n=== PER-CATEGORY FLOW PERFORMANCE & COVERAGE ===")
    print(category_summary.to_string(index=False))

    print("\n=== GLOBAL SOC TRIAGE TELEMETRY ===")
    for k, v in global_summary.items():
        print(f"  {k:<32}: {v}")

    out_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if max_rows > 0 else f"{ablation_level}_seed{seed}"
    category_summary.to_csv(out_dir / f"category_summary_{suffix}.csv", index=False)
    pd.DataFrame([global_summary]).to_csv(
        out_dir / f"global_summary_{suffix}.csv", index=False
    )

    if max_rows == 0:
        known_mask_test = split.y_test >= 0
        y_tk = split.y_test[known_mask_test]
        probs_tk = probs_test[known_mask_test]

        e4_rows = []
        for a_val in [0.10, 0.05]:
            cal_true_s = 1.0 - probs_cal[np.arange(len(split.y_cal)), split.y_cal]
            nc = len(cal_true_s)
            q_m = float(
                np.quantile(
                    cal_true_s,
                    min(1.0, np.ceil((nc + 1) * (1 - a_val)) / nc),
                    method="higher",
                )
            )
            m_sets = (1.0 - probs_tk) <= q_m
            m_covs = {
                sanitizer.inverse_category_map_[c]: float(
                    np.mean(m_sets[y_tk == c, c]) * 100
                )
                for c in np.unique(y_tk)
            }
            e4_rows.append({
                "Method": "Marginal_SCP",
                "Alpha": a_val,
                "Target_Cov": (1 - a_val) * 100,
                "Overall_Cov": round(
                    float(np.mean(m_sets[np.arange(len(y_tk)), y_tk]) * 100), 2
                ),
                "Worst_Class_Cov": round(min(m_covs.values()), 2),
                "Mean_Class_Cov": round(float(np.mean(list(m_covs.values()))), 2),
                "Mean_Set_Size": round(float(np.mean(np.sum(m_sets, axis=1))), 3),
                "Empty_Sets": int(np.sum(np.sum(m_sets, axis=1) == 0)),
            })

            mcp_a = MondrianConformalCalibrator(alpha=a_val, random_state=seed).fit(
                probs_cal, split.y_cal
            )
            mcp_sets, _, _ = mcp_a.predict_sets_and_margins(probs_tk)
            mcp_covs = {
                sanitizer.inverse_category_map_[c]: float(
                    np.mean(mcp_sets[y_tk == c, c]) * 100
                )
                for c in np.unique(y_tk)
            }
            e4_rows.append({
                "Method": "Mondrian_MCP",
                "Alpha": a_val,
                "Target_Cov": (1 - a_val) * 100,
                "Overall_Cov": round(
                    float(np.mean(mcp_sets[np.arange(len(y_tk)), y_tk]) * 100), 2
                ),
                "Worst_Class_Cov": round(min(mcp_covs.values()), 2),
                "Mean_Class_Cov": round(float(np.mean(list(mcp_covs.values()))), 2),
                "Mean_Set_Size": round(float(np.mean(np.sum(mcp_sets, axis=1))), 3),
                "Empty_Sets": int(np.sum(np.sum(mcp_sets, axis=1) == 0)),
            })
            if a_val == 0.10:
                pd.DataFrame({
                    "Category": list(m_covs.keys()),
                    "Calib_Stratum_Size": [
                        mcp_a.stratum_counts_[int(c)] for c in np.unique(y_tk)
                    ],
                    "Test_Flows": [int(np.sum(y_tk == c)) for c in np.unique(y_tk)],
                    "Marginal_SCP_Cov": [round(v, 2) for v in m_covs.values()],
                    "Mondrian_MCP_Cov": [
                        round(mcp_covs[k], 2) for k in m_covs.keys()
                    ],
                }).to_csv(out_dir / "suite_e4_per_class_coverage.csv", index=False)

        pd.DataFrame(e4_rows).to_csv(
            out_dir / "suite_e4_marginal_vs_mondrian.csv", index=False
        )

        osr_euclid = DualStageOpenSetDetector(
            distance_metric="euclidean",
            tail_size_eta=50,
            attenuation_vartheta=osr_detector.attenuation_vartheta,
            rec_quantile_theta=0.96,
        ).fit(V_train, split.y_train, y_pred_train, V_cal, split.y_cal)
        zd_prob_euclid, _ = osr_euclid.score_open_set(V_test, logits_test)

        zd_score_closed = 1.0 - np.max(probs_test, axis=1)
        zd_score_conformal_only = 1.0 - np.max(p_values_test, axis=1)
        zd_score_true_hybrid = 0.5 * zd_score_conformal_only + 0.5 * zero_day_prob

        is_true_zd = (split.y_test == -1).astype(int)
        pd.DataFrame([
            {
                "OSR_Method": "Closed_MaxProb_Baseline",
                "OpenAUROC": round(float(roc_auc_score(is_true_zd, zd_score_closed)), 4),
                "OpenAUPR": round(
                    float(average_precision_score(is_true_zd, zd_score_closed)), 4
                ),
            },
            {
                "OSR_Method": "Euclidean_OpenMax_NativeFit",
                "OpenAUROC": round(float(roc_auc_score(is_true_zd, zd_prob_euclid)), 4),
                "OpenAUPR": round(
                    float(average_precision_score(is_true_zd, zd_prob_euclid)), 4
                ),
            },
            {
                "OSR_Method": "Mahalanobis_EVT_Stage2",
                "OpenAUROC": round(float(roc_auc_score(is_true_zd, zero_day_prob)), 4),
                "OpenAUPR": round(
                    float(average_precision_score(is_true_zd, zero_day_prob)), 4
                ),
            },
            {
                "OSR_Method": "Mondrian_MaxPValue_Nonconformity",
                "OpenAUROC": round(
                    float(roc_auc_score(is_true_zd, zd_score_conformal_only)), 4
                ),
                "OpenAUPR": round(
                    float(average_precision_score(is_true_zd, zd_score_conformal_only)),
                    4,
                ),
            },
            {
                "OSR_Method": "True_Hybrid_Mahalanobis_Conformal_PVal",
                "OpenAUROC": round(
                    float(roc_auc_score(is_true_zd, zd_score_true_hybrid)), 4
                ),
                "OpenAUPR": round(
                    float(average_precision_score(is_true_zd, zd_score_true_hybrid)), 4
                ),
            },
        ]).to_csv(out_dir / "suite_e3_osr_comparison.csv", index=False)

        with open(out_dir / f"run_manifest_seed{seed}.json", "w") as mf:
            json.dump(
                {
                    "audit_log": sanitizer.audit_log_,
                    "retained_features": sanitizer.feature_columns_,
                    "dropped_features": sanitizer.dropped_columns_,
                    "split_metadata": split.metadata,
                    "ks_fallback_classes": mondrian_cp.ks_fallback_activated_classes_,
                    "tuned_vartheta": osr_detector.attenuation_vartheta,
                    "tuned_tau_hi": tuned_tau,
                    "global_summary": global_summary,
                },
                mf,
                indent=2,
            )

    print(f"\nCompleted in {time.time() - t0:.2f}s. Artifacts saved to {out_dir}.")


def run_cross_dataset_ablation(
    unsw_path: str,
    cic_path: str,
    cic_sample: int = 200000,
    seed: int = 42,
    fixed_data_seed: int = 42,
) -> None:
    print("=" * 68)
    print(
        f"RUNNING SUITE E1 & E2: CROSS-DATASET ABLATION | Model Seed: {seed} |"
        f" Fixed Data Seed: {fixed_data_seed}"
    )
    print("=" * 68)
    df_unsw = pd.read_parquet(unsw_path)
    df_cic = pd.read_parquet(cic_path)
    if len(df_cic) > cic_sample:
        df_cic = (
            df_cic.sample(n=cic_sample, random_state=fixed_data_seed)
            .sort_index()
            .reset_index(drop=True)
        )

    results = []
    for level in ["Baseline", "Set_A", "Set_B", "Set_C"]:
        t_level = time.time()
        sanitizer = ForensicDataSanitizer(
            ablation_level=level, label_col="Label", binary_col="Label"
        )
        u_clean = sanitizer.clean_and_harmonize(df_unsw, is_train_reference=True)
        c_clean = sanitizer.clean_and_harmonize(df_cic, is_train_reference=False)

        X_u = np.nan_to_num(
            u_clean[sanitizer.feature_columns_].to_numpy(dtype=np.float64),
            nan=0.0,
            posinf=1e15,
            neginf=-1e15,
        ).astype(np.float32)
        y_u = u_clean["Label"].to_numpy(dtype=np.int64)

        X_c = np.nan_to_num(
            c_clean[sanitizer.feature_columns_].to_numpy(dtype=np.float64),
            nan=0.0,
            posinf=1e15,
            neginf=-1e15,
        ).astype(np.float32)
        y_c = c_clean["Label"].to_numpy(dtype=np.int64)

        split_u, scaler_u = create_temporal_four_way_split(
            X_u, y_u, deduplicate_train_only=True
        )
        backbone_u = HybridNIDSBackbone(
            feature_names=sanitizer.feature_columns_, random_state=seed
        )
        backbone_u.fit(split_u.X_train, split_u.y_train)

        pred_u_in = backbone_u.tree_backbone.predict(split_u.X_test)
        f1_u_in = f1_score(split_u.y_test, pred_u_in, average="macro")

        X_c_scaled = np.clip(scaler_u.transform(X_c), -1e6, 1e6)
        pred_u2c = backbone_u.tree_backbone.predict(X_c_scaled)
        f1_u2c = f1_score(y_c, pred_u2c, average="macro")
        bal_u2c = balanced_accuracy_score(y_c, pred_u2c)
        mcc_u2c = matthews_corrcoef(y_c, pred_u2c)

        split_c, scaler_c = create_temporal_four_way_split(
            X_c, y_c, deduplicate_train_only=True
        )
        backbone_c = HybridNIDSBackbone(
            feature_names=sanitizer.feature_columns_, random_state=seed
        )
        backbone_c.fit(split_c.X_train, split_c.y_train)

        X_u_scaled = np.clip(scaler_c.transform(X_u), -1e6, 1e6)
        pred_c2u = backbone_c.tree_backbone.predict(X_u_scaled)
        f1_c2u = f1_score(y_u, pred_c2u, average="macro")
        bal_c2u = balanced_accuracy_score(y_u, pred_c2u)

        print(
            f"  [{level:<8}] ({len(sanitizer.feature_columns_)} feats) in"
            f" {time.time() - t_level:.1f}s | UNSW->CIC18 F1: {f1_u2c:.4f} |"
            f" CIC18->UNSW F1: {f1_c2u:.4f}"
        )

        results.append({
            "Schema": level,
            "Features": len(sanitizer.feature_columns_),
            "UNSW_InDist_F1": round(f1_u_in, 4),
            "UNSW_to_CIC18_F1": round(f1_u2c, 4),
            "U2C_BalAcc": round(bal_u2c, 4),
            "U2C_MCC": round(mcc_u2c, 4),
            "CIC18_to_UNSW_F1": round(f1_c2u, 4),
            "C2U_BalAcc": round(bal_c2u, 4),
        })

    res_df = pd.DataFrame(results)
    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    res_df.to_csv(
        f"artifacts/metrics/cross_dataset_ablation_seed{seed}.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke_test",
        choices=["smoke_test", "full_unsw", "cross_eval"],
    )
    parser.add_argument("--ablation", type=str, default="Set_C")
    parser.add_argument("--zero_day", type=str, default="Worms")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    unsw_p = cfg["paths"]["unsw_v2"]
    cic_p = cfg["paths"]["cic18_v2"]

    if args.mode == "smoke_test":
        run_multiclass_pipeline(
            unsw_p,
            ablation_level=args.ablation,
            zero_day_family=args.zero_day,
            max_rows=100000,
            seed=args.seed,
        )
    elif args.mode == "full_unsw":
        run_multiclass_pipeline(
            unsw_p,
            ablation_level=args.ablation,
            zero_day_family=args.zero_day,
            max_rows=0,
            seed=args.seed,
        )
    elif args.mode == "cross_eval":
        run_cross_dataset_ablation(unsw_p, cic_p, seed=args.seed)
