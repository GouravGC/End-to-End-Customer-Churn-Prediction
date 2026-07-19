import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from src.logger import logger
from src.exception import CustomException
from src.utils import evaluate_models, save_object

from src.constants.paths import BEST_MODEL_PATH


class ModelTrainer:

    def __init__(self):
        pass

    def initiate_model_trainer(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:

            logger.info("=" * 70)
            logger.info("MODEL TRAINING STARTED")
            logger.info("=" * 70)

            # Local import for optional dependency to avoid import-time errors
            try:
                from xgboost import XGBClassifier
            except Exception as e:
                XGBClassifier = None

            models = {

                "Logistic Regression": LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),

                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),

                "Random Forest": RandomForestClassifier(
                    random_state=42
                ),

                "Extra Trees": ExtraTreesClassifier(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),

                "AdaBoost": AdaBoostClassifier(
                    random_state=42
                ),

                "XGBoost": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                ) if XGBClassifier is not None else None
            }

            params = {

                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10]
                },

                "Decision Tree": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10]
                },

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None]
                },

                "Extra Trees": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1, 1],
                    "n_estimators": [50, 100]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "n_estimators": [100, 200]
                }
            }

            # Remove models that are None (optional dependencies not installed)
            filtered_models = {k: v for k, v in models.items() if v is not None}
            filtered_params = {k: params[k] for k in filtered_models.keys()}

            report, trained_models = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                filtered_models,
                filtered_params
            )

            best_model_name = max(report, key=report.get)

            best_model = trained_models[best_model_name]

            save_object(
                BEST_MODEL_PATH,
                best_model
            )

            logger.info(
                f"Best Model : {best_model_name}"
            )

            logger.info(
                f"ROC AUC : {report[best_model_name]:.4f}"
            )

            logger.info("=" * 70)
            logger.info("MODEL TRAINING COMPLETED")
            logger.info("=" * 70)

            return best_model

        except Exception as e:

            raise CustomException(e, sys)