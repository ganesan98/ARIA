import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

INPUT = "data/conn.log.labeled"
os.makedirs("output", exist_ok=True)

# These are the columns in the Zeek/Bro conn.log format
COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes",
    "history", "orig_pkts", "orig_ip_bytes", "resp_pkts",
    "resp_ip_bytes", "tunnel_parents", "label", "detailed_label"
]

# These are the features we'll actually use for classification
FEATURE_COLS = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"
]

def preprocess():
    print("Loading data...")
    df = pd.read_csv(
        INPUT,
        sep="\t",
        comment="#",
        names=COLUMNS,
        low_memory=False
    )

    print(f"Raw rows: {len(df)}")

    # Replace missing values marked as '-' with NaN then drop
    df.replace("-", np.nan, inplace=True)
    df.dropna(subset=FEATURE_COLS + ["label"], inplace=True)

    # Convert features to numeric
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=FEATURE_COLS, inplace=True)

    # Simplify labels: Benign vs Malicious
    df["binary_label"] = df["label"].apply(
        lambda x: 0 if "Benign" in str(x) else 1
    )

    print(f"Clean rows: {len(df)}")
    print(f"Label distribution:\n{df['binary_label'].value_counts()}")

    X_raw = df[FEATURE_COLS].values.astype(float)
    y     = df["binary_label"].values

    # Fitting the scaler on all data (train + test) leaks test statistics into
    # training — this inflates accuracy.  The scaler must only see training data.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)   # learn mean/std from train only
    X_test  = scaler.transform(X_test_raw)        # apply same params to test

    # Save everything — preprocess now produces the canonical train/test split
    np.save("output/X_train.npy", X_train)
    np.save("output/X_test.npy",  X_test)
    np.save("output/y_train.npy", y_train)
    np.save("output/y_test.npy",  y_test)
    joblib.dump(scaler,       "output/scaler.pkl")
    joblib.dump(FEATURE_COLS, "output/feature_cols.pkl")

    print("Saved X_train.npy, X_test.npy, y_train.npy, y_test.npy, scaler.pkl")

if __name__ == "__main__":
    preprocess()