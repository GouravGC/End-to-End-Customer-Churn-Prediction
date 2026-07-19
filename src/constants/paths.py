import os


# Project Root Directory
PROJECT_ROOT = os.getcwd()


# Artifacts Directory
ARTIFACTS_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts"
)


# Dataset Directories
DATASET_DIR = os.path.join(
    ARTIFACTS_DIR,
    "datasets"
)


RAW_DATA_DIR = os.path.join(
    DATASET_DIR,
    "raw"
)


PROCESSED_DATA_DIR = os.path.join(
    DATASET_DIR,
    "processed"
)


EXTERNAL_DATA_DIR = os.path.join(
    DATASET_DIR,
    "external"
)


# Dataset Files
RAW_DATA_PATH = os.path.join(
    RAW_DATA_DIR,
    "raw_dataset.csv"
)


PROCESSED_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "processed_dataset.csv"
)

# Working / cleaned / model dataset paths used by DataProcessing
WORKING_DATA_PATH = os.path.join(
    ARTIFACTS_DIR,
    "processed",
    "working_dataset.csv"
)

CLEANED_DATA_PATH = os.path.join(
    ARTIFACTS_DIR,
    "processed",
    "cleaned_dataset.csv"
)

MODEL_DATA_PATH = os.path.join(
    ARTIFACTS_DIR,
    "processed",
    "model_dataset.csv"
)


# Model Directory
MODEL_DIR = os.path.join(
    ARTIFACTS_DIR,
    "models"
)


# Model File Path
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)


# Preprocessor Path
PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

# Backwards-compatible names expected by components
BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

# Label encoder path used by transformation/prediction
LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

REPORT_DIR = os.path.join(
    ARTIFACTS_DIR,
    "reports"
)


FEATURE_ENGINEERING_REPORT_DIR = os.path.join(
    REPORT_DIR,
    "feature_engineering"
)


DATA_CLEANING_REPORT_PATH = os.path.join(
    FEATURE_ENGINEERING_REPORT_DIR,
    "data_cleaning_report.csv"
)