# =========================
# DVC preprocess stage
# =========================

import argparse
import json
from pathlib import Path

import pandas as pd


def run_preprocess(args):
    """
    Perform lightweight preprocessing validation and missing-value handling.

    This does not replace the Data Lead's preprocessing module.
    It exists only to make the DVC pipeline reproducible for the training flow.
    """
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    if args.target_column not in train_df.columns:
        raise ValueError(f"Target column '{args.target_column}' missing from train file.")

    if args.target_column not in test_df.columns:
        raise ValueError(f"Target column '{args.target_column}' missing from test file.")

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Train and test columns do not match.")

    train_missing_before = int(train_df.isna().sum().sum())
    test_missing_before = int(test_df.isna().sum().sum())

    for column in train_df.columns:
        if column == args.target_column:
            continue

        if pd.api.types.is_numeric_dtype(train_df[column]):
            median_value = train_df[column].median()
            train_df[column] = train_df[column].fillna(median_value)
            test_df[column] = test_df[column].fillna(median_value)

        else:
            mode_value = train_df[column].mode(dropna=True)

            if not mode_value.empty:
                fill_value = mode_value.iloc[0]
            else:
                fill_value = "missing"

            train_df[column] = train_df[column].fillna(fill_value)
            test_df[column] = test_df[column].fillna(fill_value)

    train_output_path = Path(args.train_output_path)
    test_output_path = Path(args.test_output_path)
    summary_path = Path(args.summary_path)

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    summary = {
        "target_column": args.target_column,
        "train_missing_before": train_missing_before,
        "test_missing_before": test_missing_before,
        "train_missing_after": int(train_df.isna().sum().sum()),
        "test_missing_after": int(test_df.isna().sum().sum()),
        "train_output_path": str(train_output_path),
        "test_output_path": str(test_output_path),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("Preprocess stage completed.")
    print(summary)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", default="data/splits/train.csv")
    parser.add_argument("--test-path", default="data/splits/test.csv")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--train-output-path", default="data/processed/train_processed.csv")
    parser.add_argument("--test-output-path", default="data/processed/test_processed.csv")
    parser.add_argument("--summary-path", default="reports/pipeline/preprocess_summary.json")

    return parser.parse_args()


if __name__ == "__main__":
    run_preprocess(parse_args())
