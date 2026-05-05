import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless + stateful feature engineering transformer.
    Safe for sklearn pipelines + joblib serialization.
    """

    def __init__(self, params):
        self.params = params
        self.stats_ = None

    def fit(self, X, y=None):
        fe = self.params["stat_features"]
        q = fe["high_value_quantile"]

        self.stats_ = {
            "high_value_threshold": float(X["MonthlyCharges"].quantile(q)),
            "quantile_used": q,
            "n_rows": len(X),
        }

        return self

    def transform(self, X):
        fe = self.params["feature_engineering"]
        df = X.copy()

        # -------------------------
        # engineered features
        # -------------------------
        df["AvgMonthlyCharges"] = np.where(
            df["tenure"] > 0,
            df["TotalCharges"] / df["tenure"],
            df["MonthlyCharges"],
        )

        df["ChargeDeviation"] = df["MonthlyCharges"] - df["AvgMonthlyCharges"]
        df["TotalCharges_log"] = np.log1p(df["TotalCharges"])

        df["NumServices"] = df[fe["service_cols"]].eq("Yes").sum(axis=1)

        threshold = fe["new_customer_tenure_threshold"]

        df["IsNewCustomer"] = (df["tenure"] <= threshold).astype(int)

        df["IsRiskySegment"] = (
            (df["Contract"] == fe["risky_contract_value"]) &
            (df["tenure"] <= threshold)
        ).astype(int)

        # -------------------------
        # stat feature (from fit only)
        # -------------------------
        df["IsHighValueCustomer"] = (
            df["MonthlyCharges"] >= self.stats_["high_value_threshold"]
        ).astype(int)

        return df