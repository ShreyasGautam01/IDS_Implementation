# src/generate_paper_artifacts.py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
import yaml

from src.conformal_mondrian import MondrianConformalCalibrator
from src.data_forensics import ForensicDataSanitizer
from src.models import HybridNIDSBackbone
from src.open_set_evt import DualStageOpenSetDetector
from src.partitioning import create_temporal_four_way_split


def aggregate_four_seed_cross_eval(metrics_dir: Path) -> pd.DataFrame:
    """Aggregates cross_dataset_ablation_seed42..45.csv into Mean and Std statistics."""
    seed_files = sorted(metrics_dir.glob("cross_dataset_ablation_seed*.csv"))
    if not seed_files:
        raise FileNotFoundError("No cross_dataset_ablation_seed*.csv files found.")

    dfs = []
    for f in seed_files:
        df_s = pd.read_csv(f)
        df_s["Source_File"] = f.name
        dfs.append(df_s)

    combined = pd.concat(dfs, ignore_index=True)
    metric_cols = [
        "UNSW_InDist_F1",
        "UNSW_to_CIC18_F1",
        "U2C_BalAcc",
        "U2C_MCC",
        "CIC18_to_UNSW_F1",
        "C2U_BalAcc",
    ]

    summary_rows = []
    for schema, grp in combined.groupby("Schema", sort=False):
        row = {
            "Schema": schema,
            "Features": int(grp["Features"].iloc[0]),
            "Seeds": len(grp),
        }
        for col in metric_cols:
            row[f"{col}_Mean"] = round(float(grp[col].mean()), 4)
            row[f"{col}_Std"] = round(float(grp[col].std()), 4)
            row[f"{col}_Median"] = round(float(grp[col].median()), 4)
            row[f"{col}_Min"] = round(float(grp[col].min()), 4)
            row[f"{col}_Max"] = round(float(grp[col].max()), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        metrics_dir / "cross_dataset_4seed_summary.csv", index=False
    )
    return summary_df


def run_supplementary_suites_and_plots(
    config_path: str = "configs/dataset_config.yaml", seed: int = 42
) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    metrics_dir = Path("artifacts/metrics")
    figs_dir = Path("artifacts/figures")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("GENERATING 4-SEED SUMMARY, CALIBRATED BASELINES & PAPER FIGURES")
    print("=" * 68)

    # -------------------------------------------------------------------------
    # STEP 1: Aggregate 4-Seed Cross-Dataset Ablation (Suites E1 & E2)
    # -------------------------------------------------------------------------
    print("[1/5] Aggregating 4-Seed Cross-Dataset Ablation (Seeds 42-45)...")
    c_summary_df = aggregate_four_seed_cross_eval(metrics_dir)
    display_cols = [
        "Schema",
        "Features",
        "Seeds",
        "UNSW_InDist_F1_Mean",
        "UNSW_to_CIC18_F1_Mean",
        "UNSW_to_CIC18_F1_Std",
        "U2C_BalAcc_Mean",
        "CIC18_to_UNSW_F1_Mean",
        "CIC18_to_UNSW_F1_Std",
    ]
    print(c_summary_df[display_cols].to_string(index=False))

    # Load stratified sample of UNSW-v2 for baseline ablation tables
    df_raw = pd.read_parquet(cfg["paths"]["unsw_v2"])
    df_attack = df_raw[df_raw["Label"] == 1]
    df_benign = df_raw[df_raw["Label"] == 0].sample(n=100000, random_state=seed)
    df_sub = (
        pd.concat([df_benign, df_attack]).sort_index().reset_index(drop=True)
    )

    sanitizer = ForensicDataSanitizer(
        ablation_level="Set_C", label_col="Attack", binary_col="Label"
    )
    df_clean = sanitizer.clean_and_harmonize(df_sub, is_train_reference=True)
    X_all, y_all = sanitizer.encode_categories(
        df_clean, zero_day_families=["Worms"]
    )
    split, _ = create_temporal_four_way_split(
        X_all, y_all, ratios=(0.55, 0.15, 0.15, 0.15)
    )

    backbone = HybridNIDSBackbone(
        feature_names=sanitizer.feature_columns_, random_state=seed
    )
    backbone.fit(split.X_train, split.y_train)

    probs_cal = backbone.predict_proba(split.X_cal)
    known_test_mask = split.y_test >= 0
    X_test_known = split.X_test[known_test_mask]
    y_test_known = split.y_test[known_test_mask]
    probs_test_known = backbone.predict_proba(X_test_known)

    # -------------------------------------------------------------------------
    # STEP 2: Suite E4 Marginal Split Conformal (SCP) vs. Mondrian (MCP)
    # -------------------------------------------------------------------------
    print("\n[2/5] Evaluating Suite E4: Marginal SCP vs. Mondrian MCP...")
    e4_rows = []
    for alpha in [0.10, 0.05]:
        cal_true_scores = 1.0 - probs_cal[np.arange(len(split.y_cal)), split.y_cal]
        n_cal = len(cal_true_scores)
        q_marg = float(
            np.quantile(
                cal_true_scores,
                min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal),
                method="higher",
            )
        )
        marg_sets = (1.0 - probs_test_known) <= q_marg

        marg_class_cov = {
            sanitizer.inverse_category_map_[c]: float(
                np.mean(marg_sets[y_test_known == c, c]) * 100
            )
            for c in np.unique(y_test_known)
        }
        e4_rows.append({
            "Method": "Marginal_SCP",
            "Alpha": alpha,
            "Target_Cov": (1 - alpha) * 100,
            "Overall_Cov": round(
                float(
                    np.mean(
                        marg_sets[np.arange(len(y_test_known)), y_test_known]
                    )
                    * 100
                ),
                2,
            ),
            "Worst_Class_Cov": round(min(marg_class_cov.values()), 2),
            "Mean_Class_Cov": round(
                float(np.mean(list(marg_class_cov.values()))), 2
            ),
            "Mean_Set_Size": round(
                float(np.mean(np.sum(marg_sets, axis=1))), 3
            ),
            "Empty_Sets": int(np.sum(np.sum(marg_sets, axis=1) == 0)),
        })

        mcp = MondrianConformalCalibrator(alpha=alpha, random_state=seed).fit(
            probs_cal, split.y_cal
        )
        mcp_sets, _, _ = mcp.predict_sets_and_margins(probs_test_known)
        mcp_class_cov = {
            sanitizer.inverse_category_map_[c]: float(
                np.mean(mcp_sets[y_test_known == c, c]) * 100
            )
            for c in np.unique(y_test_known)
        }
        e4_rows.append({
            "Method": "Mondrian_MCP",
            "Alpha": alpha,
            "Target_Cov": (1 - alpha) * 100,
            "Overall_Cov": round(
                float(
                    np.mean(
                        mcp_sets[np.arange(len(y_test_known)), y_test_known]
                    )
                    * 100
                ),
                2,
            ),
            "Worst_Class_Cov": round(min(mcp_class_cov.values()), 2),
            "Mean_Class_Cov": round(
                float(np.mean(list(mcp_class_cov.values()))), 2
            ),
            "Mean_Set_Size": round(float(np.mean(np.sum(mcp_sets, axis=1))), 3),
            "Empty_Sets": int(np.sum(np.sum(mcp_sets, axis=1) == 0)),
        })

        if alpha == 0.10:
            per_class_cp_df = pd.DataFrame({
                "Category": list(marg_class_cov.keys()),
                "Marginal_SCP_Cov": [
                    round(v, 2) for v in marg_class_cov.values()
                ],
                "Mondrian_MCP_Cov": [
                    round(mcp_class_cov[k], 2) for k in marg_class_cov.keys()
                ],
            })
            per_class_cp_df.to_csv(
                metrics_dir / "suite_e4_per_class_coverage.csv", index=False
            )

    e4_df = pd.DataFrame(e4_rows)
    e4_df.to_csv(metrics_dir / "suite_e4_marginal_vs_mondrian.csv", index=False)
    print(e4_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 3: Suite E3 Unified GBDT-Backbone Open-Set Recognition Comparison
    # -------------------------------------------------------------------------
    print("\n[3/5] Evaluating Suite E3: Unified GBDT-Backbone OSR Comparison...")
    # Extract raw GBDT decision-function logits so all OSR methods share the exact same tree backbone
    logits_train_gbdt = backbone.tree_backbone.decision_function(split.X_train)
    logits_cal_gbdt = backbone.tree_backbone.decision_function(split.X_cal)
    logits_test_gbdt = backbone.tree_backbone.decision_function(split.X_test)
    probs_all_test = backbone.predict_proba(split.X_test)

    # Shift logits to positive activation domain for Weibull OpenMax attenuation
    V_train_gbdt = logits_train_gbdt - np.min(logits_train_gbdt, axis=1, keepdims=True) + 1e-3
    V_cal_gbdt = logits_cal_gbdt - np.min(logits_cal_gbdt, axis=1, keepdims=True) + 1e-3
    V_test_gbdt = logits_test_gbdt - np.min(logits_test_gbdt, axis=1, keepdims=True) + 1e-3

    y_pred_train = np.argmax(backbone.predict_proba(split.X_train), axis=1)

    osr_maha = DualStageOpenSetDetector(
        tail_size_eta=50, attenuation_vartheta=0.85, rec_quantile_theta=0.96
    )
    osr_maha.fit(V_train_gbdt, split.y_train, y_pred_train, V_cal_gbdt, split.y_cal)
    zd_prob_maha, _ = osr_maha.score_open_set(V_test_gbdt, V_test_gbdt)

    osr_euclid = DualStageOpenSetDetector(
        tail_size_eta=50, attenuation_vartheta=0.85, rec_quantile_theta=0.96
    )
    osr_euclid.fit(V_train_gbdt, split.y_train, y_pred_train, V_cal_gbdt, split.y_cal)
    for c in osr_euclid.known_classes_:
        osr_euclid.inv_covs_[c] = np.eye(V_train_gbdt.shape[1])
    zd_prob_euclid, _ = osr_euclid.score_open_set(V_test_gbdt, V_test_gbdt)

    zd_score_closed = 1.0 - np.max(probs_all_test, axis=1)
    # Hybrid ensemble score combining closed-set probability margin and Mahalanobis tail probability
    zd_score_hybrid = 0.5 * zd_score_closed + 0.5 * zd_prob_maha

    is_true_zd = (split.y_test == -1).astype(int)
    e3_df = pd.DataFrame([
        {
            "OSR_Method": "Closed_MaxProb_Baseline",
            "OpenAUROC": round(float(roc_auc_score(is_true_zd, zd_score_closed)), 4),
            "OpenAUPR": round(float(average_precision_score(is_true_zd, zd_score_closed)), 4),
        },
        {
            "OSR_Method": "Euclidean_OpenMax_GBDT",
            "OpenAUROC": round(float(roc_auc_score(is_true_zd, zd_prob_euclid)), 4),
            "OpenAUPR": round(float(average_precision_score(is_true_zd, zd_prob_euclid)), 4),
        },
        {
            "OSR_Method": "Mahalanobis_EVT_GBDT",
            "OpenAUROC": round(float(roc_auc_score(is_true_zd, zd_prob_maha)), 4),
            "OpenAUPR": round(float(average_precision_score(is_true_zd, zd_prob_maha)), 4),
        },
        {
            "OSR_Method": "Hybrid_Mahalanobis_Conformal_OSR",
            "OpenAUROC": round(float(roc_auc_score(is_true_zd, zd_score_hybrid)), 4),
            "OpenAUPR": round(float(average_precision_score(is_true_zd, zd_score_hybrid)), 4),
        },
    ])
    e3_df.to_csv(metrics_dir / "suite_e3_osr_comparison.csv", index=False)
    print(e3_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 4: Suite E5 Boundary-Drift Sensitivity Sweep
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating Suite E5: Constrained Feature Sensitivity Sweep...")
    y_bin_train = (split.y_train > 0).astype(int)
    y_bin_test = (y_test_known > 0).astype(int)

    # Inject 2% high-volume benign file-transfer profile so unconstrained trees overfit upper tail
    rng = np.random.default_rng(seed)
    X_bin_train = split.X_train.copy()
    vol_indices = [
        i
        for i, col in enumerate(sanitizer.feature_columns_)
        if any(k in col.upper() for k in ("BYTES", "PKTS", "THROUGHPUT"))
    ]
    benign_tr_idx = np.where(y_bin_train == 0)[0]
    elephant_idx = rng.choice(benign_tr_idx, size=int(0.05 * len(benign_tr_idx)), replace=False)
    X_bin_train[np.ix_(elephant_idx, vol_indices)] += 6.0

    clf_monotone = HybridNIDSBackbone(
        feature_names=sanitizer.feature_columns_, random_state=seed
    )
    clf_monotone.fit(X_bin_train, y_bin_train)

    clf_unconstrained = HybridNIDSBackbone(
        feature_names=sanitizer.feature_columns_, random_state=seed
    )
    clf_unconstrained.tree_backbone.set_params(monotonic_cst=None)
    clf_unconstrained.tree_backbone.fit(X_bin_train, y_bin_train)

    e5_rows = []
    for eps in [0.00, 0.25, 0.50, 0.75, 1.00]:
        X_shift = X_test_known.copy()
        atk_mask = y_bin_test == 1
        X_shift[np.ix_(atk_mask, vol_indices)] += eps * 6.0

        pred_u = clf_unconstrained.tree_backbone.predict(X_shift)
        pred_m = clf_monotone.tree_backbone.predict(X_shift)

        e5_rows.append({
            "Epsilon": eps,
            "Unconstrained_Attack_Recall": round(
                float(recall_score(y_bin_test, pred_u) * 100), 2
            ),
            "Monotone_Attack_Recall": round(
                float(recall_score(y_bin_test, pred_m) * 100), 2
            ),
            "Unconstrained_MacroF1": round(
                float(f1_score(y_bin_test, pred_u, average="macro")), 4
            ),
            "Monotone_MacroF1": round(
                float(f1_score(y_bin_test, pred_m, average="macro")), 4
            ),
        })

    e5_df = pd.DataFrame(e5_rows)
    e5_df.to_csv(metrics_dir / "suite_e5_constrained_evasion.csv", index=False)
    print(e5_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 5: Render Publication Figures (With 4-Seed Error Bars on Fig 1)
    # -------------------------------------------------------------------------
    print("\n[5/5] Rendering Publication Figures to artifacts/figures/...")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    x = np.arange(len(c_summary_df))
    w = 0.22
    ax.bar(
        x - 1.5 * w,
        c_summary_df["UNSW_InDist_F1_Mean"],
        yerr=c_summary_df["UNSW_InDist_F1_Std"],
        capsize=4,
        width=w,
        label="UNSW In-Dist Macro-F1",
        color="#2b5c8f",
    )
    ax.bar(
        x - 0.5 * w,
        c_summary_df["UNSW_to_CIC18_F1_Mean"],
        yerr=c_summary_df["UNSW_to_CIC18_F1_Std"],
        capsize=4,
        width=w,
        label="Zero-Shot UNSW -> CIC18 F1",
        color="#d95f02",
    )
    ax.bar(
        x + 0.5 * w,
        c_summary_df["U2C_BalAcc_Mean"],
        yerr=c_summary_df["U2C_BalAcc_Std"],
        capsize=4,
        width=w,
        label="Zero-Shot UNSW -> CIC18 BalAcc",
        color="#7570b3",
    )
    ax.bar(
        x + 1.5 * w,
        c_summary_df["CIC18_to_UNSW_F1_Mean"],
        yerr=c_summary_df["CIC18_to_UNSW_F1_Std"],
        capsize=4,
        width=w,
        label="Zero-Shot CIC18 -> UNSW F1",
        color="#1b9e77",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.Schema}\n({r.Features} feats)" for _, r in c_summary_df.iterrows()]
    )
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("Score (Mean ± Std across 4 Seeds)")
    ax.set_title(
        "Suite E1 & E2: 4-Seed Bidirectional Cross-Dataset Leakage Ablation"
    )
    ax.legend(loc="upper right", frameon=True, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig1_cross_domain_ablation.png", dpi=300)
    fig.savefig(figs_dir / "fig1_cross_domain_ablation.pdf")
    plt.close(fig)

    print("All 4-seed tables and figures updated successfully in artifacts/!")


if __name__ == "__main__":
    run_supplementary_suites_and_plots()