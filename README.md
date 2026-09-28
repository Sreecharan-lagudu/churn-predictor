# 📉 Customer Churn Predictor — Explainable AI

> **Will this customer leave? And WHY?** An end-to-end machine learning project that predicts telecom customer churn and explains every individual prediction with SHAP.

![Demo](docs/demo.gif)

## ✨ What it does

Type in a customer's contract, charges, tenure… and get a churn probability **plus a ranked chart of what's driving that specific prediction** (e.g. "month-to-month contract pushes risk up, 2-year tenure pulls it down") — so a retention team knows not just *who* is at risk, but *what to do about it*.

## 🎯 Why it matters

Acquiring a new customer costs 5–25× more than retaining one. Predicting *who* will leave is only half the job — the business needs to know *why*. This project delivers both: a tuned classifier and per-customer explanations.

## 📊 Results

Trained on the IBM Telco Customer Churn dataset (7,043 customers, 19 features). Three models compared by cross-validated AUC:

| Model | AUC (cross-validated) |
|---|---|
| **Logistic Regression (selected)** | **0.845** (test: 0.842) |
| Random Forest | 0.824 |
| XGBoost | 0.835 |

**Key churn drivers found (SHAP):** month-to-month contracts, fiber-optic internet, high monthly charges, short tenure, electronic-check payments.

## 🚀 Run it (no installation needed)

No Python installed? Use GitHub Codespaces: click the green **Code** button → **Codespaces** tab → *Create codespace*. A full environment opens in your browser, then run these in its terminal:

    pip install -r requirements.txt
    python -m src.download_data
    python -m src.train
    streamlit run app.py

## 📦 Data

IBM Telco Customer Churn (Kaggle, CC BY 4.0 — credit: IBM). Downloaded automatically by the script above — no Kaggle account needed.

## 🗂 Project structure

    ├── app.py                  # Streamlit app with SHAP explanations
    ├── src/preprocess.py       # loading, cleaning, feature preprocessing
    ├── src/train.py            # model comparison, evaluation, artifacts
    ├── src/download_data.py    # fetches the dataset (no Kaggle login)
    ├── notebooks/01_eda.ipynb  # exploratory data analysis
    └── data/                   # telco_churn.csv (auto-downloaded)

## 🧠 What I learned / roadmap

- [x] Handling dirty columns (TotalCharges secretly stored as strings)
- [x] Class imbalance → balanced class weights + AUC (not accuracy) as the metric
- [x] Model-agnostic explanations with SHAP
- [ ] Hyperparameter search (RandomizedSearchCV)
- [ ] Cost-sensitive threshold tuning for retention campaigns

## 📄 License

MIT
