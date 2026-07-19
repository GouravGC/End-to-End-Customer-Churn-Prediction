import os
import sys

import pandas as pd

from src.logger import logger
from src.exception import CustomException

from src.config import ProjectConfig


class DataValidation:

    def __init__(self):

        self.config = ProjectConfig()

    def validate_dataset(self, df: pd.DataFrame):

        logger.info("=" * 70)
        logger.info("DATA VALIDATION STARTED")
        logger.info("=" * 70)

        try:

            validation_report = {}

            # ---------------------------------
            # Shape
            # ---------------------------------

            validation_report["Rows"] = df.shape[0]
            validation_report["Columns"] = df.shape[1]

            # ---------------------------------
            # Missing Values
            # ---------------------------------

            validation_report["Missing Values"] = (
                df.isnull().sum().sum()
            )

            # ---------------------------------
            # Duplicate Rows
            # ---------------------------------

            validation_report["Duplicate Rows"] = (
                df.duplicated().sum()
            )

            # ---------------------------------
            # Target Column
            # ---------------------------------

            validation_report["Target Column Exists"] = (
                self.config.target_column in df.columns
            )

            # ---------------------------------
            # Data Types
            # ---------------------------------

            validation_report["Total Numerical Columns"] = (
                df.select_dtypes(include=["number"]).shape[1]
            )

            validation_report["Total Categorical Columns"] = (
                df.select_dtypes(include=["object"]).shape[1]
            )

            # ---------------------------------
            # Save Validation Report
            # ---------------------------------

            report_df = pd.DataFrame(
                validation_report,
                index=[0]
            )

            os.makedirs(
                "artifacts/reports",
                exist_ok=True
            )

            report_df.to_csv(
                "artifacts/reports/data_validation_report.csv",
                index=False
            )

            logger.info("Validation Completed Successfully.")

            logger.info("=" * 70)
            logger.info("DATA VALIDATION COMPLETED")
            logger.info("=" * 70)

            return validation_report

        except Exception as e:

            raise CustomException(e, sys)