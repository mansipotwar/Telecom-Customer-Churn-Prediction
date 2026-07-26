<div align="center">

# 📊 Telecom Customer Churn Prediction

### An End-to-End Machine Learning System for Telecom Churn Analysis & Prediction

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</p>

<p>
  <img src="https://img.shields.io/badge/ROC--AUC-83.59%25-2E8B57?style=for-the-badge">
  <img src="https://img.shields.io/badge/Accuracy-80.38%25-4169E1?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
</p>

<p>
  <b>Analyze customer behavior.</b>
  <br>
  <b>Identify churn patterns.</b>
  <br>
  <b>Predict customer churn.</b>
</p>

</div>

---

## 📌 Project Overview

Customer churn is a major business challenge in the telecommunications industry. When customers discontinue their services, companies lose recurring revenue and must invest additional resources in acquiring new customers.

This project develops an end-to-end machine learning system that analyzes telecom customer behavior and predicts the likelihood of customer churn.

The final trained model is integrated into an interactive Streamlit application that allows users to enter a customer's profile and receive a real-time churn prediction, probability score, and risk classification.

---

## 🎯 Business Problem

The primary objective of this project is to answer:

> **Can we predict whether a telecom customer is likely to churn based on their demographic information, services, contract details, tenure, and billing behavior?**

A churn prediction system can help businesses:

- Identify customers at higher risk of churn
- Understand factors associated with customer churn
- Prioritize customer retention efforts
- Support data-driven business decisions
- Reduce potential revenue loss

---

## 📊 Dataset

The project uses a telecom customer churn dataset containing demographic, service, contract, and billing information.

### Dataset Summary

| Attribute | Value |
|---|---|
| Total Customers | 7,032 |
| Input Features | 19 |
| Target Variable | `Churn` |
| Problem Type | Binary Classification |

### Feature Categories

#### 👤 Demographics

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`

#### 📡 Services

- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

#### 📄 Contract & Tenure

- `Contract`
- `tenure`

#### 💳 Billing

- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

---

## 🔍 Exploratory Data Analysis

The exploratory analysis focused on identifying patterns and relationships between customer characteristics and churn behavior.

The analysis examined:

- Overall churn distribution
- Customer tenure
- Contract type
- Monthly charges
- Total charges
- Internet service
- Technical support
- Payment methods
- Customer service subscriptions

### Key Observations

- Month-to-month customers showed higher churn risk.
- Customers with longer tenure generally showed lower churn risk.
- Contract type was an important churn-related feature.
- Monthly charges contributed to customer churn behavior.
- Technical support availability showed a relationship with churn behavior.
- Payment method was an important customer characteristic in churn analysis.

---

## 🤖 Machine Learning Models

Two classification models were trained and evaluated.

### 1. Logistic Regression

Logistic Regression was used as the primary model because it provides:

- Probability-based predictions
- Good classification performance
- Interpretability through model coefficients
- A strong ROC-AUC score

### 2. Random Forest

Random Forest was trained as a comparison model to evaluate whether an ensemble tree-based approach improved prediction performance.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **80.38%** | **64.85%** | 57.22% | **60.80%** | **83.59%** |
| Random Forest | 76.76% | 55.35% | **64.97%** | 59.78% | 81.46% |

### 🏆 Selected Model: Logistic Regression

Logistic Regression was selected as the primary model based on its stronger overall performance across:

- Accuracy
- Precision
- F1 Score
- ROC-AUC

The trained Logistic Regression model and preprocessing pipeline were saved and integrated into the Streamlit application.

---

## 🧠 Model Interpretability

The Logistic Regression coefficients were analyzed to understand which features were associated with higher or lower churn probability.

### Features Associated with Higher Churn Risk

- Month-to-month contracts
- Fiber optic internet service
- Electronic check payment method
- Higher total charges
- No online security
- No technical support

### Features Associated with Lower Churn Risk

- Longer customer tenure
- Two-year contracts
- DSL internet service
- Lower monthly charges
- Customers without paperless billing

> These relationships represent patterns learned by the model from the dataset and should be interpreted as associations rather than direct causation.

---

## 🖥️ Streamlit Application

The trained machine learning model is integrated into an interactive Streamlit dashboard.

Users can enter:

### 👤 Customer Profile

- Gender
- Senior Citizen status
- Partner status
- Dependents

### 📡 Services

- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### 📄 Contract & Tenure

- Contract Type
- Tenure

### 💳 Billing

- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

The application then generates a live churn prediction.

---

## 🔮 Prediction Output

### 📊 Churn Probability

The model calculates the probability that the customer will churn.

### ⚠️ Risk Classification

| Churn Probability | Risk Level |
|---|---|
| Below 30% | Low Risk |
| 30% – 69% | Medium Risk |
| 70% or above | High Risk |

### Example Output

```text
Predicted Churn: Yes
Churn Probability: 79.05%
Risk Level: High Risk
```

### 🔄 Machine Learning Pipeline

Raw Customer Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Selection
        ↓
Saved Model + Preprocessor
        ↓
Streamlit Application
        ↓
Live Churn Prediction

📁 Project Structure

Telecom-Customer-Churn-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── telco_customer_churn.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── 1_eda.ipynb
│   └── 02_Machine_Learning_Model.ipynb
│
└── text

🛠️ Tech Stack
Programming Language
Python
Data Analysis
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Machine Learning
Scikit-learn
Application Development
Streamlit
Model Serialization
Joblib
Development Tools
Jupyter Notebook
VS Code
Git
GitHub

⚙️ Installation
1. Clone the repository
git clone https://github.com/mansipotwar/Telecom-Customer-Churn-Prediction.git

2. Navigate to the project directory
cd Telecom-Customer-Churn-Prediction

3. Create a virtual environment
python -m venv .venv
4. Activate the virtual environment
Windows
.venv\Scripts\activate
5. Install dependencies
pip install -r requirements.txt

▶️ Run the Application

Start the Streamlit application:

python -m streamlit run app.py

The application will open in your browser at:

http://localhost:8501

🔐 Model Integration

The application uses the saved machine learning artifacts:

models/
├── logistic_regression_model.pkl
└── preprocessor.pkl

The prediction flow is:

User Input
    ↓
Pandas DataFrame
    ↓
Saved Preprocessor
    ↓
Transformed Features
    ↓
Saved Logistic Regression Model
    ↓
Prediction + Probability
    ↓
Risk Classification

👩‍💻 Author

Mansi Potwar

🎯 Focus: Data Analytics & Machine Learning

🔗 GitHub: @mansipotwar

⭐ If you found this project interesting, feel free to explore the repository!


### Important: before you commit this README

There is **one thing we should verify**: the README says **"Logistic Regression was selected because it achieved the highest F1 Score"**, but your actual results show Random Forest recall is higher and the Logistic Regression F1 is only slightly higher. The README is generally accurate, but I recommend we check the exact model-selection reasoning before committing.

Also, your current repository has no `README.md` yet on GitHub, so after saving this file locally, run:

```bash
git status
