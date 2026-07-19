from __future__ import annotations

import pandas as pd


FEATURE_SELECTION_DECISION = {
    "CustomerID": "Remove",
    "Count": "Remove",
    "Country": "Remove",
    "State": "Remove",
    "City": "Keep",
    "Zip Code": "Keep",
    "Lat Long": "Remove",
    "Latitude": "Keep",
    "Longitude": "Keep",
    "Gender": "Keep",
    "Senior Citizen": "Keep",
    "Partner": "Keep",
    "Dependents": "Keep",
    "Tenure Months": "Keep",
    "Phone Service": "Keep",
    "Multiple Lines": "Keep",
    "Internet Service": "Keep",
    "Online Security": "Keep",
    "Online Backup": "Keep",
    "Device Protection": "Keep",
    "Tech Support": "Keep",
    "Streaming TV": "Keep",
    "Streaming Movies": "Keep",
    "Contract": "Keep",
    "Paperless Billing": "Keep",
    "Payment Method": "Keep",
    "Monthly Charges": "Keep",
    "Total Charges": "Keep",
    "Churn Label": "Target",
    "Churn Value": "Remove",
    "Churn Score": "Remove",
    "CLTV": "Remove",
    "Churn Reason": "Remove",
}


FEATURE_SELECTION_REASON = {
    "CustomerID": "Identifier",
    "Count": "Constant Column",
    "Country": "Single Unique Value",
    "State": "Single Unique Value",
    "City": "Useful Feature",
    "Zip Code": "Useful Feature",
    "Lat Long": "Duplicate of Latitude & Longitude",
    "Latitude": "Useful Numerical Feature",
    "Longitude": "Useful Numerical Feature",
    "Gender": "Useful Predictor",
    "Senior Citizen": "Useful Predictor",
    "Partner": "Useful Predictor",
    "Dependents": "Useful Predictor",
    "Tenure Months": "Useful Predictor",
    "Phone Service": "Useful Predictor",
    "Multiple Lines": "Useful Predictor",
    "Internet Service": "Useful Predictor",
    "Online Security": "Useful Predictor",
    "Online Backup": "Useful Predictor",
    "Device Protection": "Useful Predictor",
    "Tech Support": "Useful Predictor",
    "Streaming TV": "Useful Predictor",
    "Streaming Movies": "Useful Predictor",
    "Contract": "Strong Predictor",
    "Paperless Billing": "Useful Predictor",
    "Payment Method": "Useful Predictor",
    "Monthly Charges": "Useful Predictor",
    "Total Charges": "Useful Predictor",
    "Churn Label": "Target Variable",
    "Churn Value": "Duplicate Target",
    "Churn Score": "Data Leakage",
    "CLTV": "Potential Data Leakage",
    "Churn Reason": "Data Leakage",
}


REMOVE_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "Lat Long",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]


HIGH_CARDINALITY_COLUMNS = ["City", "Zip Code"]


def build_feature_selection_report(features: list[str]) -> pd.DataFrame:
    report = pd.DataFrame({"Feature": features})
    report["Decision"] = report["Feature"].map(FEATURE_SELECTION_DECISION)
    report["Reason"] = report["Feature"].map(FEATURE_SELECTION_REASON)
    return report


def apply_feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the notebook's feature selection stage for training datasets."""
    df_model = df.drop(columns=[c for c in REMOVE_COLUMNS if c in df.columns])
    df_model = df_model.drop(columns=[c for c in HIGH_CARDINALITY_COLUMNS if c in df_model.columns])
    return df_model


def split_features_target(df_model: pd.DataFrame, target_column: str = "Churn Label"):
    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]
    return X, y
