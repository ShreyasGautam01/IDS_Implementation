# src/generate_paper_artifacts.py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
import yaml

from src.data_forensics import ForensicDataSanitizer
from src.models import HybridNIDSBackbone
from src.partitioning import create_temporal_four_way_split


def aggregate_four_seed_cross_eval(metrics_dir: Path) -> pd.DataFrame:
    seed_files = sorted(metrics_dir.glob("cross_dataset_ablation_seed*.csv"))
    if not seed_files:
        raise FileNotFoundError("No cross_dataset_ablation_seed*.csv files found.")

    dfs = [pd.read_csv(f).assign(Source_File=f.name) for f in seed_files]
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
    summary_df.to_csv(metrics_dir / "cross_dataset_4seed_summary.csv", index=False)
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
    print("AGGREGATING 4-SEED SUMMARY, SUITE E5B & UNIFIED PAPER FIGURES")
    print("=" * 68)

    print("[1/3] Aggregating 4-Seed Cross-Dataset Ablation (Seeds 42-45)...")
    c_summary_df = aggregate_four_seed_cross_eval(metrics_dir)
    print(
        c_summary_df[[
            "Schema",
            "Features",
            "Seeds",
            "UNSW_InDist_F1_Mean",
            "UNSW_to_CIC18_F1_Mean",
            "UNSW_to_CIC18_F1_Std",
            "U2C_BalAcc_Mean",
            "CIC18_to_UNSW_F1_Mean",
            "CIC18_to_UNSW_F1_Std",
        ]].to_string(index=False)
    )

    print("\n[2/3] Evaluating Suite E5b: Binary Volumetric Sensitivity Sweep...")
    df_raw = pd.read_parquet(cfg["paths"]["unsw_v2"])
    df_attack = df_raw[df_raw["Label"] == 1]
    df_benign = df_raw[df_raw["Label"] == 0].sample(n=100000, random_state=seed)
    df_sub = pd.concat([df_benign, df_attack]).sort_index().reset_index(drop=True)

    sanitizer = ForensicDataSanitizer(
        ablation_level="Set_C", label_col="Attack", binary_col="Label"
    )
    df_clean = sanitizer.clean_and_harmonize(df_sub, is_train_reference=True)
    X_all, y_all = sanitizer.encode_categories(df_clean, zero_day_families=["Worms"])
    split, _ = create_temporal_four_way_split(
        X_all, y_all, ratios=(0.55, 0.15, 0.15, 0.15), deduplicate_train_only=True
    )

    known_test_mask = split.y_test >= 0
    X_test_known = split.X_test[known_test_mask]
    y_bin_train = (split.y_train > 0).astype(int)
    y_bin_test = (split.y_test[known_test_mask] > 0).astype(int)

    # Inject 5% high-volume benign file-transfer profiles in Z_t
    rng = np.random.default_rng(seed)
    X_bin_train = split.X_train.copy()
    vol_indices = [
        i
        for i, col in enumerate(sanitizer.feature_columns_)
        if any(k in col.upper() for k in ("BYTES", "PKTS", "THROUGHPUT"))
    ]
    benign_tr_idx = np.where(y_bin_train == 0)[0]
    elephant_idx = rng.choice(
        benign_tr_idx, size=int(0.05 * len(benign_tr_idx)), replace=False
    )
    X_bin_train[np.ix_(elephant_idx, vol_indices)] += 6.0

    clf_monotone = HybridNIDSBackbone(
        feature_names=sanitizer.feature_columns_, random_state=seed
    ).fit(X_bin_train, y_bin_train)

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
        fp_u = float(np.mean(pred_u[y_bin_test == 0] != 0) * 100)
        fp_m = float(np.mean(pred_m[y_bin_test == 0] != 0) * 100)

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
            "Unconstrained_Benign_FPR": round(fp_u, 2),
            "Monotone_Benign_FPR": round(fp_m, 2),
        })

    e5_df = pd.DataFrame(e5_rows)
    e5_df.to_csv(metrics_dir / "suite_e5_constrained_evasion.csv", index=False)
    print(e5_df.to_string(index=False))

    print("\n[3/3] Rendering All 3 Publication Figures from Unified Full-Run CSVs...")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    # Figure 1: 4-Seed Cross-Domain Ablation
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
    ax.set_title("Suite E1 & E2: 4-Seed Bidirectional Cross-Dataset Leakage Ablation")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig1_cross_domain_ablation.png", dpi=300)
    fig.savefig(figs_dir / "fig1_cross_domain_ablation.pdf")
    plt.close(fig)

    # Figure 2: Unified Full-Dataset Marginal SCP vs. Mondrian MCP Coverage
    per_class_cp_path = metrics_dir / "suite_e4_per_class_coverage.csv"
    if per_class_cp_path.exists():
        per_class_cp_df = pd.read_csv(per_class_cp_path)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        x = np.arange(len(per_class_cp_df))
        w = 0.36
        ax.bar(
            x - w / 2,
            per_class_cp_df["Marginal_SCP_Cov"],
            width=w,
            label="Marginal Split CP (Global q)",
            color="#e41a1c",
            alpha=0.85,
        )
        ax.bar(
            x + w / 2,
            per_class_cp_df["Mondrian_MCP_Cov"],
            width=w,
            label="Mondrian Class-Conditional CP (Per-Class q_c)",
            color="#1b9e77",
            alpha=0.9,
        )
        ax.axhline(
            90.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Nominal Target (1 - alpha = 90%)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(per_class_cp_df["Category"], rotation=25, ha="right")
        ax.set_ylabel("Empirical Coverage (%)")
        ax.set_ylim(0, 105)
        ax.set_title(
            "Suite E4: Full-Dataset Per-Class Coverage (Marginal vs. Mondrian CP)"
        )
        ax.legend(loc="lower right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(figs_dir / "fig2_marginal_vs_mondrian_coverage.png", dpi=300)
        fig.savefig(figs_dir / "fig2_marginal_vs_mondrian_coverage.pdf")
        plt.close(fig)

    # Figure 3: Unified Full-Dataset Raw Recall vs. Selective SOC Automated Accuracy
    full_cat_csv = metrics_dir / f"category_summary_Set_C_seed{seed}.csv"
    if full_cat_csv.exists():
        f_df = pd.read_csv(full_cat_csv)
        known_f = f_df[f_df["Category"] != "Zero-Day_Unknown"]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        x = np.arange(len(known_f))
        w = 0.36
        ax.bar(
            x - w / 2,
            known_f["Recall"],
            width=w,
            label="Raw Closed-Set Recall (%)",
            color="#6baed6",
        )
        ax.bar(
            x + w / 2,
            known_f["Auto_Acc"],
            width=w,
            label="Selective SOC Automated Accuracy (%)",
            color="#08519c",
        )
        ax.axhline(
            90.0,
            color="#d95f02",
            linestyle="--",
            linewidth=1.2,
            label="Mondrian 90% Target",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(known_f["Category"], rotation=25, ha="right")
        ax.set_ylabel("Accuracy / Recall (%)")
        ax.set_ylim(0, 108)
        ax.set_title(
            "Suite E5: Full-Dataset Raw Recall vs. Selective Automated Accuracy"
        )
        ax.legend(loc="lower right", frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(figs_dir / "fig3_soc_selective_triage_lift.png", dpi=300)
        fig.savefig(figs_dir / "fig3_soc_selective_triage_lift.pdf")
        plt.close(fig)

    print("All unified metrics and figures saved to artifacts/!")


if __name__ == "__main__":
    run_supplementary_suites_and_plots()
