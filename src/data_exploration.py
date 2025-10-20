import pandas as pd
import os

csv_path = "data/Spotify-2000.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found!")

data = pd.read_csv(csv_path)
print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())
print(data.head())
