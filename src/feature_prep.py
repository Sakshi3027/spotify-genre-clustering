# src/02_feature_prep.py
import pandas as pd
import os
import numpy as np

in_path = "data/Spotify-2000.csv"
out_path = "results/prepared_features.csv"

if not os.path.exists(in_path):
    raise FileNotFoundError(f"{in_path} not found. Put Spotify-2000.csv into the data/ folder.")

data = pd.read_csv(in_path)

# drop Index if present
if "Index" in data.columns:
    data = data.drop(columns=["Index"])

# map your dataset's audio columns to friendly variable names
# these are the columns I see in your file
available = data.columns.tolist()
print("Available columns:", available)

# features we want to use (only keep ones that actually exist)
candidate_features = [
    "Beats Per Minute (BPM)",
    "Loudness (dB)",
    "Liveness",
    "Valence",
    "Acousticness",
    "Speechiness",
    "Energy",
    "Danceability",
    "Length (Duration)",
    "Popularity"
]
features = [f for f in candidate_features if f in data.columns]
print("Using features:", features)

# create features dataframe and coerce to numeric (in case some are strings)
X = data[features].apply(pd.to_numeric, errors="coerce")

# show how many NaNs per column (for your inspection)
print("Missing values per feature:")
print(X.isna().sum())

# Option: drop rows with any NaN in selected features (safe simple choice)
X_clean = X.dropna(axis=0, how="any").reset_index(drop=True)
print(f"Rows before: {len(X)}, rows after dropna: {len(X_clean)}")

# Optionally you could fillna with column means instead:
# X_clean = X.fillna(X.mean())

# Save prepared features (we'll use this file in modeling)
os.makedirs("results", exist_ok=True)
X_clean.to_csv(out_path, index=False)
print("Saved prepared features to", out_path)
