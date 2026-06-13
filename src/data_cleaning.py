import numpy as np
import pandas as pd

from config import AMOUNT_COL, CAT_COLS, CSV_HEADER_ROW, FEATURE_COLS, RISK_THRESHOLDS


def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('"', '').replace(",", "").strip())
    return x


def calculate_risk_level(amount):
    if pd.isna(amount):
        return np.nan
    t1, t2, t3 = RISK_THRESHOLDS
    if amount <= t1:
        return 1
    elif amount <= t2:
        return 2
    elif amount <= t3:
        return 3
    return 4


def load_and_process_exact(filepath, header=CSV_HEADER_ROW):
    print("Loading and processing data...")
    df = pd.read_csv(filepath, header=header)
    df.columns = df.columns.str.strip()

    df[AMOUNT_COL] = df[AMOUNT_COL].apply(clean_currency)
    df["risk_level"] = df[AMOUNT_COL].apply(calculate_risk_level)

    df_clean = df.dropna(subset=FEATURE_COLS + ["risk_level"]).copy()

    for col in CAT_COLS:
        df_clean[col] = df_clean[col].astype("category").cat.codes

    X = df_clean[FEATURE_COLS].values
    y = df_clean["risk_level"].values.reshape(-1, 1)

    return X, y, FEATURE_COLS


if __name__ == "__main__":
    from config import DATA_SAMPLE, SAMPLE_HEADER_ROW

    X, y, names = load_and_process_exact(DATA_SAMPLE, header=SAMPLE_HEADER_ROW)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("features:", names)
