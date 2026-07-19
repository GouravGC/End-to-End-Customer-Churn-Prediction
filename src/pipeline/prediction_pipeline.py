import pandas as pd

from src.utils import load_object

from src.constants.paths import (
    PREPROCESSOR_PATH,
    LABEL_ENCODER_PATH,
    BEST_MODEL_PATH
)


class PredictionPipeline:

    def __init__(self):

        self.preprocessor = load_object(
            PREPROCESSOR_PATH
        )

        self.model = load_object(
            BEST_MODEL_PATH
        )

        self.label_encoder = load_object(
            LABEL_ENCODER_PATH
        )

        self.expected_columns = list(
            getattr(self.preprocessor, "feature_names_in_", [])
        )

    def _prepare_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(input_df, pd.DataFrame):
            raise TypeError("input_df must be a pandas DataFrame")

        df = input_df.copy()

        if "Churn Label" in df.columns:
            df = df.drop(columns=["Churn Label"])

        if self.expected_columns:
            missing_columns = [c for c in self.expected_columns if c not in df.columns]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns for prediction: {missing_columns}"
                )

            df = df[self.expected_columns]

        return df

    def predict(self, input_df):

        prepared_df = self._prepare_input(input_df)

        transformed = self.preprocessor.transform(prepared_df)

        prediction = self.model.predict(
            transformed
        )

        prediction = self.label_encoder.inverse_transform(
            prediction
        )

        return prediction