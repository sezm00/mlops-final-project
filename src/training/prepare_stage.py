# =========================
# DVC prepare stage
# =========================

import argparse
import json
from pathlib import Path

import pandas as pd


def run_prepare(args):
    """
    Validate that the train/test split files exist and summarize them.
    """
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    output_path = Path(args.output_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split file: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split file: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Train and test files do not have matching columns.")

    summary = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": int(train_df.shape[0]),
        "train_columns": int(train_df.shape[1]),
        "test_rows": int(test_df.shape[0]),
        "test_columns": int(test_df.shape[1]),
        "columns": train_df.columns.tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("Prepare stage completed.")
    print(summary)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", default="data/splits/train.csv")
    parser.add_argument("--test-path", default="data/splits/test.csv")
    parser.add_argument("--output-path", default="reports/pipeline/prepare_summary.json")

    return parser.parse_args()


if __name__ == "__main__":
    run_prepare(parse_args())
