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

    def predict(self, input_df):

        transformed = self.preprocessor.transform(
            input_df
        )

        prediction = self.model.predict(
            transformed
        )

        prediction = self.label_encoder.inverse_transform(
            prediction
        )

        return prediction