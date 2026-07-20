# 📉 End-to-End Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![SQL](https://img.shields.io/badge/SQL-Data%20Analysis-blue)
![Streamlit](https://img.shields.io/badge/Application-Streamlit-red)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-green)

An end-to-end Machine Learning project that predicts whether a customer is likely to **churn or continue using a service**.

This project combines **SQL-based data analysis, Exploratory Data Analysis, Machine Learning, and Streamlit deployment** to build a complete customer churn prediction system.

The complete workflow:

**Raw Data → SQL Analysis → Exploratory Data Analysis → Feature Engineering → Model Training → Model Evaluation → Artifact Generation → Prediction Pipeline → Streamlit Deployment**

---

# 🚀 Live Demo

The application is deployed using Streamlit Cloud:

🔗 **Customer Churn Prediction App:**  
https://end-to-end-customer-churn-prediction-gc.streamlit.app/

Users can enter customer details and receive a real-time prediction of whether the customer is likely to churn.

---

# 📌 Project Overview

Customer churn is one of the most important business problems faced by companies across industries.

Losing existing customers directly impacts revenue, customer lifetime value, and business growth. Machine Learning can help organizations identify customers who are likely to leave and take proactive retention actions.

This project develops a classification-based Machine Learning system that predicts customer churn using customer demographics, service details, account information, and behavioral attributes.

The trained model is integrated into a Streamlit application that provides real-time churn predictions.

---

# 🎯 Problem Statement

Build an end-to-end Machine Learning classification system that can predict whether a customer will churn based on historical customer information.

The goal is to:

- Identify high-risk customers
- Understand churn patterns
- Generate actionable business insights
- Support customer retention strategies

---

# ✨ Key Features

✅ Complete end-to-end Machine Learning pipeline  
✅ SQL-based customer data analysis  
✅ Exploratory Data Analysis (EDA)  
✅ Data visualization and reporting  
✅ Feature engineering workflow  
✅ Classification model experimentation  
✅ Model evaluation and comparison  
✅ Automated prediction pipeline  
✅ Streamlit interactive application  
✅ Cloud deployment using Streamlit Cloud  
✅ Production-style project structure  

---

# 🗄️ SQL-Based Data Analysis

SQL was used for **data analysis and business insights generation**.

The SQL workflow includes:

- Database creation
- Data loading
- Customer data exploration
- Churn analysis
- Aggregation queries
- Customer segmentation analysis
- Business reporting

SQL helped in understanding customer behavior and identifying important patterns before building the Machine Learning model.

**Note:**  
SQL is used for analytical purposes only. The final churn prediction is performed using the trained Machine Learning pipeline.

---

# 📊 Dataset Information

The dataset contains customer-level information along with churn labels.

## Feature Categories

| Category | Examples |
|---|---|
| Customer Information | Demographic attributes |
| Account Information | Tenure, contract details |
| Service Information | Subscribed services |
| Billing Information | Charges, payment details |
| Target Variable | Customer churn status |

---

# 🎯 Target Variable

The model predicts:

| Value | Meaning |
|---|---|
| 0 | Customer will not churn |
| 1 | Customer is likely to churn |

---

# 🏗️ Project Architecture

```
End-to-End Customer Churn Prediction
│
├── artifacts/
│   │
│   ├── raw/
│   │   └── dataset
│   │
│   ├── database/
│   │   └── SQL database files
│   │
│   ├── plots/
│   │   └── EDA visualizations
│   │
│   ├── reports/
│   │   └── Analysis reports
│   │
│   ├── model/
│   │   └── trained model artifacts
│   │
│   └── preprocessing/
│       └── preprocessing objects
│
├── notebooks/
│   └── Exploratory Analysis & Model Training
│
├── src/
│   │
│   ├── components/
│   │
│   ├── pipelines/
│   │
│   └── utilities/
│
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

# 🔄 Machine Learning Workflow

## 1. Data Collection & Ingestion

Steps performed:

- Load raw customer dataset
- Store raw data
- Prepare data for analysis

---

## 2. Exploratory Data Analysis

Performed analysis on:

- Customer distribution
- Churn distribution
- Feature relationships
- Customer behavior patterns
- Important churn factors

Generated plots and reports are stored inside the artifacts folder.

---

## 3. Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Categorical feature encoding
- Numerical feature transformation
- Feature scaling
- Data preparation for model training

Techniques used:

- Pandas
- NumPy
- Scikit-learn preprocessing utilities

---

## 4. Model Training

Multiple classification algorithms were evaluated.

Models experimented with:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

The best-performing model was selected based on evaluation metrics.

---

# 📈 Model Evaluation

Since this is a classification problem, multiple evaluation metrics were considered.

## Accuracy

Measures overall prediction correctness.

## Precision

Measures how many predicted churn customers were actually churn customers.

## Recall

Measures how many actual churn customers were correctly identified.

## F1 Score

Balances precision and recall.

## ROC-AUC Score

Measures the model's ability to distinguish between churn and non-churn customers.

---

# 🏆 Model Performance

| Metric | Score |
|---|---|
| Accuracy | Add Result |
| Precision | Add Result |
| Recall | Add Result |
| F1 Score | Add Result |
| ROC-AUC | Add Result |

---

# 🖥️ Streamlit Application

The Streamlit application provides an interactive interface for customer churn prediction.

Workflow:

```
User Input
     |
     ↓
Data Preprocessing Pipeline
     |
     ↓
Trained ML Model
     |
     ↓
Prediction Result
```

The application provides:

- Customer churn prediction
- Prediction output
- Real-time inference

---

# 📦 Model Artifacts

The project stores generated artifacts including:

```
artifacts/

├── raw dataset
├── SQL database
├── trained model
├── preprocessing objects
├── plots
└── reports
```

These artifacts allow reproducibility and deployment of the trained Machine Learning pipeline.

---

# 🛠️ Tech Stack

## Programming

- Python

## Data Analysis

- Pandas
- NumPy
- SQL

## Machine Learning

- Scikit-learn
- XGBoost

## Visualization

- Matplotlib
- Seaborn

## Application

- Streamlit

## Deployment

- Streamlit Cloud

## Development Tools

- Git
- GitHub
- VS Code

---

# ⚙️ Installation & Setup

Clone the repository:

```bash
git clone https://github.com/GouravGC/End-to-End-Customer-Churn-Prediction.git
```

Navigate into the project:

```bash
cd End-to-End-Customer-Churn-Prediction
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit application:

```bash
streamlit run app.py
```

---

# 📸 Application Screenshots

Add screenshots:

```
screenshots/

├── home.png
├── input.png
└── prediction.png
```

Recommended screenshots:

- Streamlit homepage
- Customer input form
- Prediction output

---

# 🌐 Deployment Architecture

```
Developer
     |
     ↓
GitHub Repository
     |
     ↓
Streamlit Cloud
     |
     ↓
Streamlit Application
     |
     ↓
Prediction Pipeline
     |
     ↓
Customer Churn Prediction
```

---

# 🔮 Future Improvements

Future enhancements:

- Add SHAP-based explainability
- Add customer risk scoring
- Add retention recommendation system
- Add MLflow experiment tracking
- Add Docker deployment
- Add CI/CD pipeline
- Add automated model monitoring
- Add model retraining workflow

---

# 👨‍💻 Author

**Your Name**

Machine Learning Engineer | Data Science Enthusiast | AI Developer

🔗 GitHub:  
https://github.com/GouravGC

🔗 LinkedIn:  
https://linkedin.com/in/yourprofile

---

# ⭐ Support

If you found this project useful, consider giving this repository a ⭐ on GitHub.
