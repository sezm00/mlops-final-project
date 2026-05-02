# =========================
# Placeholder data preparation
# =========================

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def create_placeholder_data(
    train_path="data/splits/train.csv",
    test_path="data/splits/test.csv",
    target_column="target",
):
    """
    Create placeholder train/test splits only if real split files do not exist.
    This prevents the DVC pipeline from failing before the real dataset is ready.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)

    if train_path.exists() and test_path.exists():
        print("Train/test split files already exist. Placeholder data was not created.")
        return

    train_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
    df[target_column] = y

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_column],
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Placeholder train/test data created.")
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")


if __name__ == "__main__":
    create_placeholder_data()
