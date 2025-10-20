# src/03_modeling_kmeans.py
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend to avoid blocking
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

in_path = "results/prepared_features.csv"
out_path_csv = "results/features_with_clusters.csv"
out_path_fig = "results/figures/k_selection.png"

if not os.path.exists(in_path):
    raise FileNotFoundError(f"{in_path} not found. Run src/02_feature_prep.py first.")

# Load features
X = pd.read_csv(in_path)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find best k using Elbow and Silhouette
Ks = range(2, 13)
inertia = []
silhouette = []

for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels))

# Plot and save
os.makedirs("results/figures", exist_ok=True)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(Ks, inertia, '-o')
plt.xlabel('k')
plt.title('Elbow: Inertia vs k')

plt.subplot(1,2,2)
plt.plot(Ks, silhouette, '-o')
plt.xlabel('k')
plt.title('Silhouette Score vs k')

plt.tight_layout()
plt.savefig(out_path_fig, dpi=150)
plt.close()  # close the figure to avoid blocking

# Choose k (example: 5). You can adjust after looking at the plot.
k = 5
km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km_final.fit_predict(X_scaled)

# Attach cluster labels and save
X_out = X.copy()
X_out['cluster'] = labels
X_out.to_csv(out_path_csv, index=False)

print("KMeans clustering done.")
print("Cluster results saved to:", out_path_csv)
print("Elbow/Silhouette figure saved to:", out_path_fig)
