import os
import sys

import sqlite3
import pandas as pd

from src.logger import logger
from src.exception import CustomException

from src.constants.paths import (
    RAW_DATA_PATH
)


class DataIngestion:

    def __init__(self):

        self.raw_data_path = RAW_DATA_PATH

    def initiate_data_ingestion(self):

        logger.info("=" * 70)
        logger.info("DATA INGESTION STARTED")
        logger.info("=" * 70)

        try:

            if not os.path.exists(self.raw_data_path):

                raise FileNotFoundError(
                    f"{self.raw_data_path} not found."
                )

            df = pd.read_csv(
                self.raw_data_path
            )

            logger.info(
                f"Dataset Loaded Successfully."
            )

            logger.info(
                f"Dataset Shape : {df.shape}"
            )

            logger.info("=" * 70)
            logger.info("DATA INGESTION COMPLETED")
            logger.info("=" * 70)

            return df

        except Exception as e:

            raise CustomException(e, sys)