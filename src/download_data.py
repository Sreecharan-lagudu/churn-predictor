"""Download the IBM Telco Customer Churn dataset (no Kaggle login needed).

Usage:
    python -m src.download_data
"""
import pathlib

import pandas as pd

URL = ("https://raw.githubusercontent.com/IBM/"
       "telco-customer-churn-on-icp-4d/master/Telco-Customer-Churn.csv")


def main():
    out = pathlib.Path("data")
    out.mkdir(exist_ok=True)
    print("Downloading dataset (~1 MB)...")
    df = pd.read_csv(URL)
    df.to_csv(out / "telco_churn.csv", index=False)
    print(f"Saved data/telco_churn.csv - {len(df):,} customers, "
          f"{df.shape[1]} columns")


if __name__ == "__main__":
    main()
