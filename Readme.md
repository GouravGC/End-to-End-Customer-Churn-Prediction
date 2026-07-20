# Customer Churn Prediction (SQL-First)

## Overview

This project predicts whether a customer is likely to churn using supervised machine learning. The project follows a production-ready machine learning workflow, including data ingestion, validation, preprocessing, feature engineering, model training, model persistence, and deployment.

The project is built using a modular architecture instead of a single Jupyter Notebook, making it easier to maintain, test, and deploy.

---

## Problem Statement

Customer churn is one of the biggest challenges faced by subscription-based businesses. The objective of this project is to identify customers who are likely to leave so that proactive retention strategies can be implemented.

---

## Project Features

* SQL-first data processing workflow
* Modular machine learning architecture
* Data validation
* Data cleaning and preprocessing
* Feature engineering
* Automatic model training
* Model evaluation
* Artifact generation
* Prediction pipeline
* Interactive Streamlit web application

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── artifacts/
├── notebooks/
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── constants/
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── app.py
├── main.py
├── setup.py
├── requirements.txt
├── README.md
└── Dockerfile
```

---

## Machine Learning Workflow

1. Data Ingestion
2. Data Validation
3. Data Processing
4. Feature Engineering
5. Data Transformation
6. Model Training
7. Model Evaluation
8. Model Saving
9. Prediction Pipeline
10. Streamlit Deployment

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SQLAlchemy
* Streamlit
* Matplotlib
* Seaborn
* SHAP
* Joblib

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Customer-Churn-Prediction
```

Create a virtual environment or Conda environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the project:

```bash
pip install -e .
```

---

## Training

To train the complete pipeline:

```bash
python main.py
```

---

## Running the Application

```bash
streamlit run app.py
```

---

## Generated Artifacts

The training pipeline automatically generates:

* best_model.pkl
* preprocessor.pkl
* label_encoder.pkl
* processed datasets
* validation reports

---

## Future Improvements

* Docker support
* CI/CD pipeline
* MLflow experiment tracking
* DVC data versioning
* Cloud deployment
* Model monitoring
* Automated retraining

---

## Author

**Gourav Chhatwani**

Machine Learning | Data Science | AI Engineering
