from src.logger import logger

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_processing import DataProcessor
from src.components.data_transformation import DataTransformation


class TrainPipeline:

    def __init__(self):
        logger.info("Training Pipeline Initialized")

    def run_pipeline(self):

        logger.info("========== TRAINING PIPELINE STARTED ==========")

        # Step 1
        ingestion = DataIngestion()
        raw_df = ingestion.initiate_data_ingestion()

        # Step 2
        validator = DataValidation()
        validator.validate_data(raw_df)

        # Step 3
        processor = DataProcessor()
        processed_df = processor.process_data(raw_df)

        # Step 4
        transformer = DataTransformation()
        transformer.initiate_data_transformation(processed_df)

        logger.info("========== TRAINING PIPELINE COMPLETED ==========")