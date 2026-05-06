import pandas as pd
import great_expectations as ge


def validate_reference_data(df: pd.DataFrame):

    # Convert properly using Dataset API
    df_ge = ge.dataset.PandasDataset(df)

    # -------------------------
    # Schema expectations
    # -------------------------
    df_ge.expect_column_to_exist("tenure")
    df_ge.expect_column_to_exist("MonthlyCharges")
    df_ge.expect_column_to_exist("TotalCharges")
    df_ge.expect_column_to_exist("Churn")

    # -------------------------
    # Business rules
    # -------------------------
    df_ge.expect_column_values_to_be_between("tenure", 0, 100)
    df_ge.expect_column_values_to_be_between("MonthlyCharges", 0, 200)
    df_ge.expect_column_values_to_be_between("TotalCharges", 0, 10000)

    df_ge.expect_column_values_to_be_in_set("Churn", [0, 1])

    # -------------------------
    # Feature checks
    # -------------------------
    df_ge.expect_column_values_to_be_in_set("IsNewCustomer", [0, 1])
    df_ge.expect_column_values_to_be_in_set("IsHighValueCustomer", [0, 1])

    df_ge.expect_column_values_to_be_between("NumServices", 0, 7)

    # -------------------------
    # Final validation
    # -------------------------
    result = df_ge.validate()

    if not result.success:
        raise ValueError("Great Expectations validation FAILED")

    return result