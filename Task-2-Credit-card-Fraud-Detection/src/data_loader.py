"""
data_loader.py

This is the bit that grabs the data. I tried to make it work with the Sparkov fraud dataset 
(the big simulated one from Kaggle that everyone seems to use for these projects). 
If the CSV isn't there (which it probably won't be for most people), it spits out some 
synthetic stuff that roughly matches the shape and fraud rate so you can still run everything.

I cobbled this together over a weekend because the original Kaggle file is huge and 
I didn't want to assume everyone has it downloaded. The synthetic version isn't perfect 
but it's good enough to test the pipeline without crashing.

Real dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# Just some paths so I don't have to hardcode stuff everywhere
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "fraud_data.csv"  # drop your fraudTrain.csv here and rename it if you want


def load_data(path: str | Path = CSV_PATH, verbose: bool = True) -> pd.DataFrame:
    """Tries to load the real CSV. If it's not there, generates fake data instead.
    Prints a quick summary so you know what's going on.
    """
    path = Path(path)

    if path.exists():
        if verbose:
            print(f"[data] Found the file at {path}, loading it up...")
        df = pd.read_csv(path)
        # Sometimes the column is called 'Class' instead of 'is_fraud', so I added this quick fix
        if "is_fraud" not in df.columns and "Class" in df.columns:
            df = df.rename(columns={"Class": "is_fraud"})
    else:
        if verbose:
            print(
                "[data] No fraud_data.csv found in the data folder. "
                "Spinning up some synthetic data instead so the code doesn't explode.\n"
                "   (You can grab the real dataset from Kaggle and drop it here if you want real numbers)"
            )
        df = _make_synthetic(n_samples=100000, fraud_rate=0.005, seed=42)

    if verbose:
        fraud_pct = df["is_fraud"].mean() * 100
        print(
            f"[data] Loaded {len(df):,} rows  |  {int(df['is_fraud'].sum())} fraud cases  |  "
            f"{fraud_pct:.3f}% fraud rate"
        )

    return df


def _make_synthetic(n_samples: int = 100000, fraud_rate: float = 0.005, seed: int = 42) -> pd.DataFrame:
    """I built this fake data generator because I was too lazy to download the full dataset every time.
    It creates columns that look like the Sparkov one and tries to make fraud cases a bit more 
    'suspicious' (higher amounts, etc.). It's not scientifically perfect but it gets the job done
    for testing the rest of the pipeline.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_samples * fraud_rate)

    # I just picked some reasonable ranges after staring at the real data for a bit
    df = pd.DataFrame({
        "trans_date_trans_time": pd.date_range(start="2020-01-01", periods=n_samples, freq="min").strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": rng.integers(100000000000000, 999999999999999, size=n_samples, dtype=np.int64),
        "merchant": [f"merchant_{i}" for i in rng.integers(0, 500, size=n_samples)],
        "category": np.random.choice(
            ["grocery_pos", "shopping_pos", "food_dining", "travel", "entertainment", 
             "gas_transport", "misc_net", "shopping_net"], 
            n_samples
        ),
        "amt": np.abs(rng.normal(loc=50, scale=80, size=n_samples)),
        "first": [f"first_{i}" for i in rng.integers(0, 100, size=n_samples)],
        "last": [f"last_{i}" for i in rng.integers(0, 100, size=n_samples)],
        "gender": np.random.choice(["M", "F"], n_samples),
        "street": [f"street_{i}" for i in rng.integers(0, 200, size=n_samples)],
        "city": [f"city_{i}" for i in rng.integers(0, 100, size=n_samples)],
        "state": np.random.choice(["CA", "NY", "TX", "FL", "PA", "IL", "OH", "MI"], n_samples),
        "zip": rng.integers(10000, 99999, size=n_samples),
        "lat": rng.uniform(25.0, 49.0, size=n_samples),
        "long": rng.uniform(-125.0, -70.0, size=n_samples),
        "city_pop": rng.integers(500, 500000, size=n_samples),
        "job": np.random.choice(["engineer", "teacher", "nurse", "retail", "driver", "manager", "student", "other"], n_samples),
        "dob": pd.date_range(start="1950-01-01", periods=n_samples, freq="D").strftime("%Y-%m-%d")[rng.permutation(n_samples)],
        "trans_num": [f"trans_{i:010d}" for i in rng.integers(0, 10000000000, size=n_samples)],
        "unix_time": rng.integers(1577836800, 1609459200, size=n_samples),
        "merch_lat": rng.uniform(25.0, 49.0, size=n_samples),
        "merch_long": rng.uniform(-125.0, -70.0, size=n_samples),
    })

    # Make the fraud rows a bit more 'obvious' - higher amounts and random other tweaks
    fraud_indices = rng.choice(n_samples, n_fraud, replace=False)
    df["is_fraud"] = 0
    df.loc[fraud_indices, "is_fraud"] = 1
    # fraud tends to be bigger transactions (at least in my experience looking at these datasets)
    df.loc[fraud_indices, "amt"] = df.loc[fraud_indices, "amt"] * rng.uniform(3, 8, size=n_fraud)
    df["amt"] = df["amt"].clip(lower=0.01)

    # shuffle it so it's not in any particular order
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)