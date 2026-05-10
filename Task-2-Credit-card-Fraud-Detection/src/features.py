"""
features.py

I spent way too much time figuring out what features actually help with fraud detection 
on this Sparkov dataset. The raw data has a ton of columns (names, addresses, etc.) that 
aren't super useful on their own, so I cooked up some new ones: time stuff, distance 
between buyer and merchant (fraudsters often operate far from their 'home'), age, 
amount bins, and some quick label encoding for the categories.

Nothing revolutionary, but it seems to help the models a bit. I kept it simple because 
I didn't want to over-engineer things for this internship task.
"""

import numpy as np
import pandas as pd


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Quick haversine formula I copied from StackOverflow once and never changed.
    Gives distance in km between two lat/long points. Good enough for this.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes the raw dataframe and adds a bunch of features I thought might be useful.
    Returns a new dataframe with the extra columns. I tried to keep it readable.
    """
    out = df.copy()

    # Amount is super skewed, so log it. I added the clip just in case there are zeros or negatives
    out["log_amt"] = np.log1p(out["amt"].clip(lower=0))

    # Time features - fraud often happens at weird hours or on weekends
    out["hour"] = (out["unix_time"] % 86400) // 3600
    out["day_of_week"] = pd.to_datetime(out["trans_date_trans_time"]).dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # Distance between customer's home and the merchant location.
    # I think this is a strong signal - people usually shop near home, fraudsters don't care.
    out["distance_km"] = haversine(out["lat"], out["long"], out["merch_lat"], out["merch_long"])

    # Rough age from date of birth. Older people might have different spending patterns.
    out["age"] = (pd.Timestamp.now() - pd.to_datetime(out["dob"])).dt.days / 365.25

    # Binning the amount into categories because sometimes the model likes discrete values better
    out["amt_bin"] = pd.cut(
        out["amt"],
        bins=[0, 10, 50, 200, 1000, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # Label encode the main categorical columns. Not the fanciest approach but it works fine.
    for col in ["category", "gender", "state"]:
        if col in out.columns:
            out[f"{col}_enc"] = out[col].astype("category").cat.codes

    # Log of city population because bigger cities have more transactions
    out["log_city_pop"] = np.log1p(out["city_pop"])

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Figures out which columns the model should actually use.
    Drops all the ID-like columns and the raw categoricals (we use the _enc versions instead).
    Also drops the target so we don't accidentally leak it.
    """
    exclude = {
        "is_fraud", "amt", "trans_date_trans_time", "cc_num", "merchant", "first", "last",
        "street", "city", "zip", "job", "dob", "trans_num", "lat", "long", "merch_lat", "merch_long",
        "category", "gender", "state"
    }
    # Only keep columns that look numeric (int or float). This is a bit hacky but works for our case.
    return [
        c for c in df.columns 
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]