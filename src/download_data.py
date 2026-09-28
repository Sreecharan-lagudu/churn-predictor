"""Download the IBM Telco Customer Churn dataset (no Kaggle login needed).

Usage:
    python -m src.download_data
"""
import pathlib

import pandas as pd

# Public mirrors of the original IBM file (CC BY 4.0). If one is down,
# the script falls back to the next.
URLS = [
    "https://raw.githubusercontent.com/rstudio/keras-customer-churn/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/classics/customer_churn/raw_data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/carlosfab/dsnp2/master/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def main():
    out = pathlib.Path("data")
    out.mkdir(exist_ok=True)
    print("Downloading dataset (~1 MB)...")
    for url in URLS:
        try:
            df = pd.read_csv(url)
            df.to_csv(out / "telco_churn.csv", index=False)
            print(f"Saved data/telco_churn.csv - {len(df):,} customers, "
                  f"{df.shape[1]} columns")
            return
        except Exception as e:
            print(f"Mirror failed ({e.__class__.__name__}), trying next...")
    raise SystemExit("All mirrors failed - check your internet connection.")


if __name__ == "__main__":
    main()
