import numpy as np
import pandas as pd


def _clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    col = "Total Charges"
    if col in df.columns:
        # Notebook parity: replace blank-space strings, convert to numeric, then drop nulls.
        df[col] = df[col].replace(" ", np.nan)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[col])
    return df


def initiate_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic feature engineering derived from the project notebook.

    This function performs safe, idempotent transformations:
    - Creates a feature-engineering working copy
    - Replaces blank-space values in `Total Charges`
    - Converts `Total Charges` to numeric
    - Drops rows where `Total Charges` is missing after conversion

    Returns a DataFrame with the same columns and order as the input dataframe.
    """
    df = df.copy()

    df = _clean_total_charges(df)

    return df
