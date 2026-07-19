import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline

# -------------------------------
# Load Model Dataset
# -------------------------------

df = pd.read_csv(
    "artifacts/processed/model_dataset.csv"
)

# Remove target column
X = df.drop(columns=["Churn Label"])

# Take first customer
sample = X.iloc[[0]]

# -------------------------------
# Prediction
# -------------------------------

pipeline = PredictionPipeline()

prediction = pipeline.predict(sample)

print("=" * 50)
print("Prediction : ", prediction[0])
print("=" * 50)