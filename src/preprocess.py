"""Data loading and preprocessing for the Telco churn dataset."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data(path: str = "data/telco_churn.csv") -> pd.DataFrame:
    """Load raw data, fix dtypes, encode target.

    Notes
    -----
    TotalCharges is stored as a string in the raw CSV and contains ~11
    blank values. We coerce it to numeric and fill the gaps with
    MonthlyCharges * tenure (a sound estimate of the true total).
    """
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isna()
    df.loc[missing, "TotalCharges"] = (
        df.loc[missing, "MonthlyCharges"] * df.loc[missing, "tenure"]
    )

    # Binary target: 1 = churned, 0 = stayed
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df.drop(columns=["customerID"])


def get_feature_lists(df: pd.DataFrame):
    """Return (numeric, categorical) lists present in the dataframe."""
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical


def build_preprocessor(numeric, categorical) -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
        ],
        sparse_threshold=0.0,  # dense output -> easier for SHAP
    )
