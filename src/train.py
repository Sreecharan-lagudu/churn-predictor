"""Train and compare models, pick the best, evaluate, save artifacts.

Usage:
    python -m src.train
"""
import json
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.preprocess import TARGET, build_preprocessor, get_feature_lists, load_data

RANDOM_STATE = 42
MODELS = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "xgboost": XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=4,
        eval_metric="logloss", random_state=RANDOM_STATE,
    ),
}


def main():
    df = load_data("data/telco_churn.csv")
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    numeric, categorical = get_feature_lists(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric, categorical)
    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_te_proc = preprocessor.transform(X_te)
    feature_names = list(preprocessor.get_feature_names_out())

    # ---- Compare models with 5-fold cross-validated AUC ----
    print(f"\n{'Model':<16}{'CV AUC (mean)':>16}{'CV AUC (std)':>14}")
    print("-" * 46)
    scores = {}
    for name, model in MODELS.items():
        cv = cross_val_score(model, X_tr_proc, y_tr, cv=5, scoring="roc_auc")
        scores[name] = cv.mean()
        print(f"{name:<16}{cv.mean():>16.4f}{cv.std():>14.4f}")

    best_name = max(scores, key=scores.get)
    print(f"\nBest model: {best_name} (CV AUC = {scores[best_name]:.4f})")

    # ---- Fit the best model on the full training set ----
    model = MODELS[best_name]
    model.fit(X_tr_proc, y_tr)

    # ---- Hold-out test evaluation ----
    proba = model.predict_proba(X_te_proc)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_auc = roc_auc_score(y_te, proba)
    print(f"\nHold-out test AUC: {test_auc:.4f}\n")
    print(classification_report(y_te, preds, target_names=["Stayed", "Churned"]))

    # ---- Save artifacts for the Streamlit app ----
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "model_name": best_name,
            "feature_names": feature_names,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "test_auc": test_auc,
        },
        "artifacts/model.joblib",
    )
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"model": best_name, "test_auc": round(test_auc, 4)}, f, indent=2)
    print("Saved artifacts/model.joblib - now run: streamlit run app.py")


if __name__ == "__main__":
    main()
