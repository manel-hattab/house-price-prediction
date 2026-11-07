# backend/preprocessing.py
import pandas as pd
import numpy as np
import json

# ── Chargement des artefacts fixes
with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

with open("models/iqr_bounds.json") as f:
    IQR_BOUNDS = json.load(f)

# ── Colonnes numériques (celles qui ont des bornes IQR)
NUM_COLS = list(IQR_BOUNDS.keys())


def cap_outliers_iqr(df):

    for col in NUM_COLS:
        if col in df.columns:
            lower = IQR_BOUNDS[col]["lower"]
            upper = IQR_BOUNDS[col]["upper"]
            df[col] = df[col].clip(lower, upper)
    return df


def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # 1. Écrêtage IQR (identique à cap_outliers_iqr du notebook)
    df = cap_outliers_iqr(df)

    # 2. Imputation médiane pour les NaN numériques restants
    for col in NUM_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(IQR_BOUNDS[col].get("median", 0))

    # 3. One-Hot Encoding (drop_first=True identique au notebook cell 75)
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 4. Aligner sur les colonnes d'entraînement (cell 75 : align join='left')
    df_encoded = df_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df_encoded


def preprocess_for_ann(data: dict, scaler) -> np.ndarray:
    df = preprocess_input(data)
    return scaler.transform(df.astype(np.float32))