# =========================
# DVC featurize stage
# =========================

import argparse
import json
from pathlib import Path

import pandas as pd


def run_featurize(args):
    """
    Save feature-ready train/test files and feature metadata.
    """
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    if args.target_column not in train_df.columns:
        raise ValueError(f"Target column '{args.target_column}' missing from train file.")

    if args.target_column not in test_df.columns:
        raise ValueError(f"Target column '{args.target_column}' missing from test file.")

    feature_columns = [
        column for column in train_df.columns
        if column != args.target_column
    ]

    train_output_path = Path(args.train_output_path)
    test_output_path = Path(args.test_output_path)
    metadata_path = Path(args.metadata_path)

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    metadata = {
        "target_column": args.target_column,
        "feature_columns": feature_columns,
        "number_of_features": len(feature_columns),
        "train_output_path": str(train_output_path),
        "test_output_path": str(test_output_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Featurize stage completed.")
    print(metadata)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", default="data/processed/train_processed.csv")
    parser.add_argument("--test-path", default="data/processed/test_processed.csv")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--train-output-path", default="data/features/train_features.csv")
    parser.add_argument("--test-output-path", default="data/features/test_features.csv")
    parser.add_argument("--metadata-path", default="reports/pipeline/feature_metadata.json")

    return parser.parse_args()


if __name__ == "__main__":
    run_featurize(parse_args())
