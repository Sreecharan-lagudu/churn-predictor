"""Customer Churn Predictor - Streamlit app with per-customer SHAP explanations.

Usage:
    streamlit run app.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load("artifacts/model.joblib")
    if bundle["model_name"] == "logreg":
        explainer = shap.LinearExplainer(
            bundle["model"],
            np.zeros((1, len(bundle["feature_names"]))),
        )
    else:
        explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer


bundle, explainer = load_artifacts()
model = bundle["model"]
preprocessor = bundle["preprocessor"]
feature_names = bundle["feature_names"]

# ---------------------------------------------------------------- UI
st.title("📉 Customer Churn Predictor")
st.caption(
    f"Model: {bundle['model_name']} · Hold-out test AUC: "
    f"{bundle['test_auc']:.3f} · Explainability: SHAP"
)

st.sidebar.header("Customer profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
internet = st.sidebar.selectbox(
    "Internet service", ["DSL", "Fiber optic", "No"]
)
payment = st.sidebar.selectbox(
    "Payment method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)",
     "Credit card (automatic)"],
)
online_security = st.sidebar.selectbox(
    "Online security", ["Yes", "No", "No internet service"]
)
tech_support = st.sidebar.selectbox(
    "Tech support", ["Yes", "No", "No internet service"]
)
senior = st.sidebar.radio("Senior citizen?", ["No", "Yes"], horizontal=True)
partner = st.sidebar.radio("Has partner?", ["No", "Yes"], horizontal=True)
dependents = st.sidebar.radio("Has dependents?", ["No", "Yes"], horizontal=True)

# ---- Build a complete input row (defaults for features not in the UI) ----
customer = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": int(senior == "Yes"),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": tech_support,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": monthly * max(tenure, 1),
        }
    ]
)

# ---------------------------------------------------------------- Prediction
X_proc = preprocessor.transform(customer)
churn_prob = float(model.predict_proba(X_proc)[0, 1])

col1, col2, col3 = st.columns(3)
col1.metric("Churn probability", f"{churn_prob:.1%}")
col2.metric(
    "Verdict", "⚠️ High risk" if churn_prob >= 0.5 else "✅ Likely to stay"
)
col3.metric(
    "Retention priority",
    "Act now" if churn_prob >= 0.7 else ("Monitor" if churn_prob >= 0.4 else "Low"),
)

st.progress(min(churn_prob, 1.0))

# ---------------------------------------------------------------- SHAP
st.subheader("Why this prediction? (SHAP values)")
shap_values = explainer.shap_values(X_proc)
if isinstance(shap_values, list):          # some explainers return a list
    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
sv = np.asarray(shap_values).reshape(-1)
if sv.size == 2 * len(feature_names):     # (classes, features) layout
    sv = sv[len(feature_names):]

pairs = sorted(zip(feature_names, sv), key=lambda p: abs(p[1]), reverse=True)[:8]
pairs = sorted(pairs, key=lambda p: p[1])  # most negative at the bottom
labels = [p[0].replace("cat__", "").replace("num__", "") for p in pairs]
vals = [p[1] for p in pairs]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
ax.barh(labels, vals, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP value  ->  pushes toward churn")
ax.set_title("Top factors for THIS customer")
st.pyplot(fig)

st.caption(
    "🔴 pushes toward churn · 🟢 pushes toward retention — "
    "built with scikit-learn / XGBoost + SHAP"
)
