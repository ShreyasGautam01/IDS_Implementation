import sys
from pathlib import Path

# Fix Windows cp1252 UnicodeEncodeError for terminal output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import polars as pl

from pathlib import Path

UNSW_PATH = Path(r"D:/Datasets/NF-UNSW-NB15-V2.parquet")
CIC18_PATH = Path(r"D:/Datasets/NF-CSE-CIC-IDS2018-V2.parquet")

def run_sanity_check(unsw_file: Path, cic_file: Path):
    print("=" * 60)
    print("PHASE 2 - PART 3: INGESTION SANITY CHECK")
    print("=" * 60)

    # 1. Feature and Column Inspection
    print("\n[1/3] Inspecting Schema and Feature Alignment...")
    df_unsw = pl.scan_parquet(unsw_file)
    df_cic = pl.scan_parquet(cic_file)

    unsw_cols = set(df_unsw.collect_schema().names())
    cic_cols = set(df_cic.collect_schema().names())

    label_cols = {"Label", "Attack"}
    shared_features = (unsw_cols & cic_cols) - label_cols
    
    print(f" -> NF-UNSW-NB15 total columns: {len(unsw_cols)}")
    print(f" -> NF-CSE-CIC-IDS2018 total columns: {len(cic_cols)}")
    print(f" -> Shared non-label features: {len(shared_features)}")

    # Diagnose the missing 2 features
    diff_unsw = unsw_cols - cic_cols
    diff_cic = cic_cols - unsw_cols
    if diff_unsw or diff_cic:
        print("\n[!] Column Schema Differences:")
        print(f" -> Columns in UNSW but NOT in CIC18: {diff_unsw}")
        print(f" -> Columns in CIC18 but NOT in UNSW: {diff_cic}")
    else:
        print(" -> All non-label columns align 1:1.")

    # 2. Check for Nulls
    print("\n[2/3] Checking for Null Values Across Datasets...")
    unsw_nulls = df_unsw.select(pl.all().null_count()).collect().to_dict(as_series=False)
    total_unsw_nulls = sum(sum(v) for v in unsw_nulls.values())

    cic_nulls = df_cic.select(pl.all().null_count()).collect().to_dict(as_series=False)
    total_cic_nulls = sum(sum(v) for v in cic_nulls.values())

    print(f" -> NF-UNSW-NB15-v2 total null cells: {total_unsw_nulls}")
    print(f" -> NF-CSE-CIC-IDS2018-v2 total null cells: {total_cic_nulls}")

    # 3. Label Distributions (Printed safely)
    print("\n[3/3] Inspecting Label Distributions...")
    print("\n-- NF-UNSW-NB15-v2 Label Distribution --")
    unsw_dist = (
        df_unsw.group_by("Label")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum() * 100).round(2).alias("percentage"))
        .collect()
    )
    for row in unsw_dist.iter_rows(named=True):
        print(f"  Class {row['Label']}: {row['count']:,} ({row['percentage']}%)")

    print("\n-- NF-CSE-CIC-IDS2018-v2 Label Distribution --")
    cic_dist = (
        df_cic.group_by("Label")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum() * 100).round(2).alias("percentage"))
        .collect()
    )
    for row in cic_dist.iter_rows(named=True):
        print(f"  Class {row['Label']}: {row['count']:,} ({row['percentage']}%)")

    print("\n" + "=" * 60)
    print("Sanity check finished.")
    print("=" * 60)

if __name__ == "__main__":
    run_sanity_check(UNSW_PATH, CIC18_PATH)