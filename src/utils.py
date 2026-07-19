import os
import sys
import json
import joblib
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logger

def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)

        logger.info(f"Object saved at {file_path}")

    except Exception as e:

        raise CustomException(e, sys)
    
def load_object(file_path):

    try:

        obj = joblib.load(file_path)

        logger.info(f"Object loaded from {file_path}")

        return obj

    except Exception as e:

        raise CustomException(e, sys)
    
def save_json(file_path, data):

        try:

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as file:

                json.dump(data, file, indent=4)

            logger.info(f"JSON saved at {file_path}")

        except Exception as e:

            raise CustomException(e, sys)
    
def load_json(file_path):

        try:

            with open(file_path, "r") as file:

                data = json.load(file)

            return data

        except Exception as e:

            raise CustomException(e, sys)
    
def save_dataframe(df, file_path):

        try:

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            df.to_csv(file_path, index=False)

            logger.info(f"DataFrame saved at {file_path}")

        except Exception as e:

            raise CustomException(e, sys)
    
def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):

        try:

            report = {}

            for model_name in models:

                model = models[model_name]

                para = param[model_name]

                gs = GridSearchCV(

                    estimator=model,

                    param_grid=para,

                    cv=5,

                    scoring="roc_auc",

                    n_jobs=-1

                )

                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)

                model.fit(X_train, y_train)

                y_prob = model.predict_proba(X_test)[:, 1]

                score = roc_auc_score(y_test, y_prob)

                report[model_name] = score

                logger.info(
                    f"{model_name} ROC AUC : {score:.4f}"
                )

            return report

        except Exception as e:

            raise CustomException(e, sys)