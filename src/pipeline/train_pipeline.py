import os

from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_processing import DataProcessing
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.feature_engineering import initiate_feature_engineering

from src.constants.paths import (
    RAW_DATA_PATH,
    ARTIFACTS_DIR,
    DATASET_DIR,
    MODEL_DIR,
    REPORT_DIR
)


class TrainPipeline:

    def __init__(self):
        self.data_ingestion = DataIngestion(raw_data_path=RAW_DATA_PATH)
        self.data_validation = DataValidation()
        self.data_processing = DataProcessing()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):

        logger.info("Starting training pipeline and ensuring artifact folders exist.")

        # Ensure artifact folders exist
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(REPORT_DIR, exist_ok=True)

        # 1) Ingest
        df = self.data_ingestion.initiate_data_ingestion()

        # 2) Validate
        self.data_validation.validate_dataset(df)

        # 3) Process / cleaning (placeholder logic lives in DataProcessing)
        processed_df = self.data_processing.process_data(df)

        # 4) Feature engineering (deterministic transformations extracted from notebook)
        engineered_df = initiate_feature_engineering(processed_df)

        # 5) Transform (encoding / scaling / split)
        (
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            preprocessor,
            label_encoder,
        ) = self.data_transformation.initiate_data_transformation(engineered_df)

        # 6) Train + tune models and persist best model
        best_model = self.model_trainer.initiate_model_trainer(
            X_train_processed,
            y_train,
            X_test_processed,
            y_test,
        )

        logger.info("Training pipeline completed.")

        return best_model