import sys
import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from src.logger import logger
from src.exception import CustomException
from src.config import ProjectConfig

from src.utils import save_object

from src.constants.paths import (
    PREPROCESSOR_PATH,
    LABEL_ENCODER_PATH
)


class DataTransformation:

    def __init__(self):

        self.config = ProjectConfig()

    def initiate_data_transformation(self, df):

        logger.info("=" * 70)
        logger.info("DATA TRANSFORMATION STARTED")
        logger.info("=" * 70)

        try:

            target_column = self.config.target_column

            X = df.drop(columns=[target_column])

            y = df[target_column]

            categorical_columns = X.select_dtypes(
                include="object"
            ).columns.tolist()

            numerical_columns = X.select_dtypes(
                exclude="object"
            ).columns.tolist()

            logger.info(
                f"Categorical Columns : {len(categorical_columns)}"
            )

            logger.info(
                f"Numerical Columns : {len(numerical_columns)}"
            )

            numeric_pipeline = Pipeline(

                steps=[
                    ("scaler", StandardScaler())
                ]

            )

            categorical_pipeline = Pipeline(

                steps=[
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]

            )

            preprocessor = ColumnTransformer(

                transformers=[

                    (
                        "num",
                        numeric_pipeline,
                        numerical_columns
                    ),

                    (
                        "cat",
                        categorical_pipeline,
                        categorical_columns
                    )

                ]

            )

            label_encoder = LabelEncoder()

            y = label_encoder.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(

                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y

            )

            X_train_processed = preprocessor.fit_transform(
                X_train
            )

            X_test_processed = preprocessor.transform(
                X_test
            )

            save_object(
                PREPROCESSOR_PATH,
                preprocessor
            )

            save_object(
                LABEL_ENCODER_PATH,
                label_encoder
            )

            logger.info("Preprocessor Saved.")

            logger.info("Label Encoder Saved.")

            logger.info("=" * 70)
            logger.info("DATA TRANSFORMATION COMPLETED")
            logger.info("=" * 70)

            return (

                X_train_processed,
                X_test_processed,
                y_train,
                y_test,
                preprocessor,
                label_encoder

            )

        except Exception as e:

            raise CustomException(e, sys)