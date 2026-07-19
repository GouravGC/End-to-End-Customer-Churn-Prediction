import sys
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.logger import logger
from src.exception import CustomException
from src.utils import save_json


class ModelEvaluation:

    def __init__(self):
        pass

    def initiate_model_evaluation(
        self,
        model,
        X_test,
        y_test
    ):

        try:

            logger.info("=" * 70)
            logger.info("MODEL EVALUATION STARTED")
            logger.info("=" * 70)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {

                "Accuracy": accuracy_score(y_test, y_pred),

                "Precision": precision_score(y_test, y_pred),

                "Recall": recall_score(y_test, y_pred),

                "F1 Score": f1_score(y_test, y_pred),

                "ROC AUC": roc_auc_score(y_test, y_prob)

            }

            logger.info(metrics)

            save_json(
                "artifacts/reports/model/model_metrics.json",
                metrics
            )

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            pd.DataFrame(report).transpose().to_csv(
                "artifacts/reports/model/classification_report.csv"
            )

            cm = pd.DataFrame(
                confusion_matrix(y_test, y_pred)
            )

            cm.to_csv(
                "artifacts/reports/model/confusion_matrix.csv",
                index=False
            )

            logger.info("=" * 70)
            logger.info("MODEL EVALUATION COMPLETED")
            logger.info("=" * 70)

            return metrics

        except Exception as e:

            raise CustomException(e, sys)