# d:\IDS_Implementation\main.py
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
import yaml

from src.conformal_mondrian import MondrianConformalCalibrator
from src.data_forensics import ForensicDataSanitizer
from src.models import HybridNIDSBackbone
from src.open_set_evt import DualStageOpenSetDetector
from src.partitioning import create_temporal_four_way_split
from src.soc_triage import SOCTriageEngine


def load_config(
    config_path: str = "configs/dataset_config.yaml",
) -> dict:
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
        f"RUNNING MULTI-CLASS OSR + MONDRIAN PIPELINE | Ablation:"
        f" {ablation_level} | Seed: {seed}"
    )
    print("=" * 68)
    t0 = time.time()

    df_raw = pd.read_parquet(parquet_path)
    if max_rows > 0 and len(df_raw) > max_rows:
        # Preserve attack representation during smoke testing
        df_attack = df_raw[df_raw["Label"] == 1]
        df_benign = df_raw[df_raw["Label"] == 0].sample(
            n=max(10000, max_rows - len(df_attack)), random_state=seed
        )
        df_raw = (
            pd.concat([df_benign, df_attack])
            .sort_index()
            .reset_index(drop=True)
        )
        print(f"[Smoke Test Mode] Using subset of {len(df_raw):,} flows.")
    else:
        print(f"[Full Mode] Loaded {len(df_raw):,} total flows.")

    # Print available multi-class Attack categories
    attack_counts = df_raw["Attack"].value_counts()
    print("\n--- Multi-Class Attack Distribution ---")
    for k, v in attack_counts.items():
        print(f"  {k:<20}: {v:>10,}")

    if zero_day_family not in attack_counts.index:
        fallback = attack_counts.index[-1]
        print(
            f"[!] '{zero_day_family}' not found; defaulting withheld Zero-Day"
            f" family to '{fallback}'"
        )
        zero_day_family = fallback

    # Stage 1: Forensic Sanitization & Feature Pruning
    sanitizer = ForensicDataSanitizer(
        ablation_level=ablation_level, label_col="Attack", binary_col="Label"
    )
    df_clean = sanitizer.clean_and_harmonize(df_raw, is_train_reference=True)
    X_all, y_all = sanitizer.encode_categories(
        df_clean, zero_day_families=[zero_day_family]
    )
    print(
        f"\n[Stage 1] Retained {len(sanitizer.feature_columns_)}"
        f" topology-invariant features under {ablation_level}."
    )

    # Stage 2: Chronological Four-Way Split
    split, _ = create_temporal_four_way_split(
        X_all, y_all, ratios=(0.55, 0.15, 0.15, 0.15)
    )
    print(
        f"[Stage 2] Split sizes -> Z_t: {len(split.y_train):,}, Z_v:"
        f" {len(split.y_val):,}, Z_c: {len(split.y_cal):,}, Z_test:"
        f" {len(split.y_test):,}"
    )

    # Stage 3: Hybrid Monotone Tree + Latent Neural Backbone
    print("[Stage 3] Fitting Hybrid Monotone GBDT + Latent Encoder on Z_t...")
    backbone = HybridNIDSBackbone(
        feature_names=sanitizer.feature_columns_, random_state=seed
    )
    backbone.fit(split.X_train, split.y_train)

    # Stage 4: Dual-Stage Mahalanobis EVT + Subspace Reconstruction OSR
    print(
        "[Stage 4] Fitting Mahalanobis Weibull EVT + Stage-2 Reconstruction"
        " Gate..."
    )
    V_train, _ = backbone.extract_penultimate_embeddings(split.X_train)
    V_cal, _ = backbone.extract_penultimate_embeddings(split.X_cal)
    V_test, logits_test = backbone.extract_penultimate_embeddings(split.X_test)

    y_pred_train = np.argmax(backbone.predict_proba(split.X_train), axis=1)
    osr_detector = DualStageOpenSetDetector(
        tail_size_eta=50, attenuation_vartheta=0.85, rec_quantile_theta=0.96
    )
    osr_detector.fit(V_train, split.y_train, y_pred_train, V_cal, split.y_cal)
    zero_day_prob, is_zero_day_rejected = osr_detector.score_open_set(
        V_test, logits_test
    )

    # Stage 5: Mondrian Class-Conditional Conformal Prediction
    print(
        f"[Stage 5] Calibrating Mondrian Conformal Quantiles on Z_c (alpha ="
        f" {alpha})..."
    )
    probs_cal = backbone.predict_proba(split.X_cal)
    probs_test = backbone.predict_proba(split.X_test)

    mondrian_cp = MondrianConformalCalibrator(alpha=alpha, random_state=seed)
    mondrian_cp.fit(probs_cal, split.y_cal)
    pred_set_mask, evidence_margin, _ = mondrian_cp.predict_sets_and_margins(
        probs_test
    )

    # Stage 6: Selective SOC Triage Routing
    print(
        f"[Stage 6] Executing Selective SOC Triage Routing (tau_hi ="
        f" {tau_hi})..."
    )
    triage = SOCTriageEngine(
        inverse_category_map=sanitizer.inverse_category_map_, tau_hi=tau_hi
    )
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
        print(f"  {k:<28}: {v}")

    out_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    category_summary.to_csv(
        out_dir / f"category_summary_{ablation_level}_seed{seed}.csv",
        index=False,
    )
    print(
        f"\nCompleted in {time.time() - t0:.2f}s. Metrics saved to {out_dir}."
    )


def run_cross_dataset_ablation(
    unsw_path: str, cic_path: str, cic_sample: int = 200000, seed: int = 42
) -> None:
    print("=" * 68)
    print(
        "RUNNING SUITE E1 & E2: BIDIRECTIONAL CROSS-DATASET LEAKAGE ABLATION"
    )
    print("=" * 68)
    df_unsw = pd.read_parquet(unsw_path)
    df_cic = pd.read_parquet(cic_path)
    if len(df_cic) > cic_sample:
        df_cic = (
            df_cic.sample(n=cic_sample, random_state=seed)
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

        # Safe conversion from float64 to float32 without overflow to inf
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

        # Direction 1: Train on UNSW-NB15-v2 -> Test In-Dist & Zero-Shot on CIC18-v2
        split_u, scaler_u = create_temporal_four_way_split(X_u, y_u)
        backbone_u = HybridNIDSBackbone(
            feature_names=sanitizer.feature_columns_, random_state=seed
        )
        backbone_u.tree_backbone.fit(split_u.X_train, split_u.y_train)

        pred_u_in = backbone_u.tree_backbone.predict(split_u.X_test)
        f1_u_in = f1_score(split_u.y_test, pred_u_in, average="macro")

        X_c_scaled = np.clip(scaler_u.transform(X_c), -1e6, 1e6)
        pred_u2c = backbone_u.tree_backbone.predict(X_c_scaled)
        f1_u2c = f1_score(y_c, pred_u2c, average="macro")
        bal_u2c = balanced_accuracy_score(y_c, pred_u2c)
        mcc_u2c = matthews_corrcoef(y_c, pred_u2c)

        # Direction 2: Train on CIC18-v2 -> Test Zero-Shot on UNSW-NB15-v2
        split_c, scaler_c = create_temporal_four_way_split(X_c, y_c)
        backbone_c = HybridNIDSBackbone(
            feature_names=sanitizer.feature_columns_, random_state=seed
        )
        backbone_c.tree_backbone.fit(split_c.X_train, split_c.y_train)

        X_u_scaled = np.clip(scaler_c.transform(X_u), -1e6, 1e6)
        pred_c2u = backbone_c.tree_backbone.predict(X_u_scaled)
        f1_c2u = f1_score(y_u, pred_c2u, average="macro")
        bal_c2u = balanced_accuracy_score(y_u, pred_c2u)

        print(
            f"  [{level:<8}] ({len(sanitizer.feature_columns_)} feats) completed"
            f" in {time.time() - t_level:.1f}s | UNSW->CIC18 F1: {f1_u2c:.4f} |"
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
    print("\n=== BIDIRECTIONAL CROSS-DATASET ABLATION RESULTS ===")
    print(res_df.to_string(index=False))
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