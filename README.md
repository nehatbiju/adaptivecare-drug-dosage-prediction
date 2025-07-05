# 💊 AdaptiveCare – Personalized Drug & Dosage Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?logo=machine-learning)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

**AdaptiveCare** is a machine learning-powered healthcare app that predicts the **most suitable drug and dosage** for a patient based on their medical history, demographics, and real-time health factors.

Developed using **XGBoost**, **Scikit-learn**, and **Streamlit**, the project combines both classification (drug label) and regression (dosage) in a two-stage ML pipeline. It provides an intuitive web interface for predictions and insights.

---

## 🚀 Features

- ✅ Predicts optimal drug from a multi-class label set
- 💊 Estimates personalized daily dosage (mg)
- 📈 Displays model performance metrics
- 📊 Shows actual vs. predicted dosage error graph
- 🧠 Feature importance visualization
- 🧪 Built with real patient dataset (simulated for privacy)

---

## 🛠 Tech Stack

- **Languages**: Python 3.11  
- **ML Libraries**: XGBoost, Scikit-learn, Pandas, NumPy  
- **Web App**: Streamlit  
- **Visualization**: Matplotlib, Seaborn  
- **Model Persistence**: joblib

---

## 🧠 ML Pipeline

- **Stage 1 (Classification)**:  
  Predict the drug label using XGBoost Classifier with stratified sampling and label encoding.

- **Stage 2 (Regression)**:  
  Predict the daily dosage using a tuned XGBoost Regressor with RMSE as the scoring metric.

---

## 📊 Model Performance

| Task | Algorithm | Metric | Score |
|------|-----------|--------|-------|
| Drug Classification | XGBoostClassifier | Accuracy | 89% |
| Dosage Prediction | XGBoostRegressor | R² Score | 0.84 |
| Dosage Prediction | XGBoostRegressor | RMSE | ±11.2 mg |

---

## 🖥️ Running the Project Locally

```bash
# Clone the repo
git clone https://github.com/nehatbiju/adaptivecare-drug-dosage-prediction.git
cd adaptivecare-drug-dosage-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
