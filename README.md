# 📊 Telecom Customer Churn Prediction

An end-to-end machine learning project that analyzes telecom customer behavior and predicts the likelihood of customer churn through an interactive Streamlit application.

---

## 🚀 Live Application

🔗 **Live Demo:** Coming Soon

---

## 📌 Project Overview

Customer churn is a major challenge in the telecommunications industry. When customers discontinue their services, companies lose recurring revenue and must spend additional resources acquiring new customers.

This project uses historical telecom customer data to:

- Analyze customer behavior and churn patterns
- Identify factors associated with customer churn
- Train machine learning models to predict churn
- Compare model performance
- Generate real-time churn predictions for individual customers
- Classify customers according to their churn risk

The final model is integrated into an interactive Streamlit application.

---

## 🎯 Business Problem

The objective of this project is to answer:

> **Can we predict whether a telecom customer is likely to churn based on their demographic information, service usage, contract details, and billing behavior?**

A predictive churn system can help businesses:

- Identify high-risk customers
- Prioritize retention efforts
- Understand important churn drivers
- Support data-driven customer retention strategies

---

## 📊 Dataset

The project uses a telecom customer churn dataset containing customer demographic, service, contract, and billing information.

### Dataset Information

- **Total Customers:** 7,032
- **Total Features:** 19 input features
- **Target Variable:** `Churn`

### Feature Categories

#### 👤 Demographics

- Gender
- Senior Citizen
- Partner
- Dependents

#### 📡 Services

- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

#### 📄 Contract & Tenure

- Contract
- Tenure

#### 💳 Billing

- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## 🔍 Exploratory Data Analysis

The analysis focused on understanding the relationship between customer characteristics and churn behavior.

Key areas analyzed included:

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
- Payment method was also an important customer characteristic in churn analysis.

---

## 🤖 Machine Learning Models

Two classification models were trained and evaluated:

### 1. Logistic Regression

Logistic Regression was selected as the primary model because it provides:

- Strong classification performance
- Probability-based predictions
- Interpretability through model coefficients

### 2. Random Forest

Random Forest was used as a comparison model to evaluate whether a tree-based ensemble approach improved prediction performance.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.38% | 64.85% | 57.22% | 60.80% | **83.59%** |
| Random Forest | 76.76% | 55.35% | **64.97%** | 59.78% | 81.46% |

### 🏆 Selected Model

**Logistic Regression**

The Logistic Regression model was selected as the primary model because it achieved the highest:

- Accuracy
- Precision
- F1 Score
- ROC-AUC

The trained model and preprocessing pipeline were saved and integrated into the Streamlit application.

---

## 🧠 Model Interpretability

The model coefficients helped identify features associated with increased or decreased churn probability.

### Features Associated with Higher Churn Risk

- Month-to-month contracts
- Fiber optic internet service
- Electronic check payment method
- Higher total charges
- Lack of online security
- Lack of technical support

### Features Associated with Lower Churn Risk

- Longer customer tenure
- Two-year contracts
- DSL internet service
- Lower monthly charges
- Customers without paperless billing

---

## 🖥️ Streamlit Application

The trained machine learning model is integrated into an interactive Streamlit dashboard.

The application allows users to enter a customer's:

- Demographic information
- Telecom service details
- Contract information
- Tenure
- Billing information

The application then generates:

### 🔮 Churn Prediction

```text
Predicted Churn: Yes / No
