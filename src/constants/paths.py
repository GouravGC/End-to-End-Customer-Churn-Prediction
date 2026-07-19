import os

ARTIFACT_DIR = "artifacts"

RAW_DATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "datasets",
    "raw",
    "raw_dataset.csv"
)

WORKING_DATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "processed",
    "working_dataset.csv"
)

CLEANED_DATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "processed",
    "cleaned_dataset.csv"
)

MODEL_DATA_PATH = os.path.join(
    ARTIFACT_DIR,
    "processed",
    "model_dataset.csv"
)

PREPROCESSOR_PATH = os.path.join(
    ARTIFACT_DIR,
    "preprocessor",
    "preprocessor.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    ARTIFACT_DIR,
    "preprocessor",
    "label_encoder.pkl"
)

BEST_MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "models",
    "best_model.pkl"
)