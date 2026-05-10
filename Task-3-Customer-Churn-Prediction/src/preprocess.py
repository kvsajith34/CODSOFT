"""
preprocess.py

Just loads the csv, drops useless columns, encodes gender and geography,
splits the data and scales it. Nothing fancy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")  # sklearn likes to complain sometimes


DROP_COLS = ["RowNumber", "CustomerId", "Surname"]  # these are just ids, no value for prediction
TARGET_COL = "Exited"


def load_data(filepath):
    # load and do basic check
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    if df.isnull().sum().sum() > 0:
        print("Warning: some missing values, but this dataset is clean usually.")
    else:
        print("No missing values, good.")

    return df


def explore_basics(df):
    print("\nQuick look at the data:")
    print(df.dtypes)
    churn_rate = df[TARGET_COL].mean() * 100
    stayed = (df[TARGET_COL] == 0).sum()
    churned = (df[TARGET_COL] == 1).sum()
    print(f"Churn rate is about {churn_rate:.1f}% ({churned} churned out of {stayed + churned})")


def preprocess(df, test_size=0.2, random_state=42):
    # drop id columns, encode cats, split, scale
    df = df.copy()

    # drop the useless id stuff
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # encode gender (0/1) and one-hot geo
    le = LabelEncoder()
    if "Gender" in df.columns:
        df["Gender"] = le.fit_transform(df["Gender"])

    if "Geography" in df.columns:
        geo_dummies = pd.get_dummies(df["Geography"], prefix="Geo", drop_first=False)
        df = pd.concat([df.drop(columns=["Geography"]), geo_dummies], axis=1)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    # stratified split so churn ratio stays the same in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # scale the numbers
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    print(f"Using {len(feature_names)} features after encoding\n")

    return X_train, X_test, y_train, y_test, feature_names, scaler
