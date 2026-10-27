#!/usr/bin/env python3
"""
CLI Pipeline Driver: Prepare and Verify Dataset Splits

Generates deterministic, stratified, and patient-grouped dataset splits
guaranteeing zero patient leakage across train, val, and test partitions.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from src.datasets.splits import DatasetSplitter
from src.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible stratified dataset splits with patient-level grouping."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to input metadata CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/splits",
        help="Directory to save split manifest CSV files.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="label",
        help="Column name for primary classification target.",
    )
    parser.add_argument(
        "--patient_col",
        type=str,
        default=None,
        help="Optional column name for unique patient IDs to enforce 0-leakage grouping.",
    )
    parser.add_argument(
        "--stratify_col",
        type=str,
        default="fitzpatrick_tone",
        help="Demographic subgroup column to stratify on (e.g. skin tone).",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train set ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        raise FileNotFoundError(f"Missing file: {metadata_path}")

    logger.info("Loading metadata from: %s", metadata_path)
    df = pd.read_csv(metadata_path)
    logger.info("Total samples: %d", len(df))

    splitter = DatasetSplitter(seed=args.seed)

    if args.patient_col and args.patient_col in df.columns:
        logger.info("Using patient-grouped partitioning on column: '%s'", args.patient_col)
        train_df, val_df, test_df = splitter.create_patient_grouped_split(
            df,
            patient_col=args.patient_col,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
    else:
        logger.info("Using stratified partitioning on: %s + [%s]", args.target_col, args.stratify_col)
        strat_cols = [args.stratify_col] if args.stratify_col in df.columns else None
        train_df, val_df, test_df = splitter.create_stratified_split(
            df,
            target_col=args.target_col,
            stratify_cols=strat_cols,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    # Audit split integrity
    audit = splitter.verify_split_integrity(
        train_df, val_df, test_df, patient_col=args.patient_col
    )
    logger.info("Split integrity audit: %s", audit)

    splits_dict = {"train": train_df, "val": val_df, "test": test_df}
    written = splitter.save_splits(splits_dict, output_dir=args.output_dir)
    logger.info("Successfully exported %d split manifests to: %s", len(written), args.output_dir)


if __name__ == "__main__":
    main()
