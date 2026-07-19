import os
import sys

import pandas as pd

from src.logger import logger
from src.exception import CustomException

from src.constants.paths import (
    WORKING_DATA_PATH,
    CLEANED_DATA_PATH,
    MODEL_DATA_PATH
)


class DataProcessing:

    def __init__(self):
        pass

    def process_data(self, df):

        logger.info("=" * 70)
        logger.info("DATA PROCESSING STARTED")
        logger.info("=" * 70)

        try:

            ###################################################
            # NOTEBOOK CLEANING LOGIC GOES HERE
            ###################################################

            # Example

            # df = df.drop(...)
            # df["Feature"] = ...

            ###################################################

            os.makedirs(
                "artifacts/processed",
                exist_ok=True
            )

            df.to_csv(
                WORKING_DATA_PATH,
                index=False
            )

            df.to_csv(
                CLEANED_DATA_PATH,
                index=False
            )

            df.to_csv(
                MODEL_DATA_PATH,
                index=False
            )

            logger.info("Processed datasets saved.")

            logger.info("=" * 70)
            logger.info("DATA PROCESSING COMPLETED")
            logger.info("=" * 70)

            return df

        except Exception as e:

            raise CustomException(e, sys)