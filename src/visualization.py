# src/04_visualization.py
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

in_path = "results/features_with_clusters.csv"
out_path_fig = "results/figures/clusters_pca.png"

if not os.path.exists(in_path):
    raise FileNotFoundError(f"{in_path} not found. Run src/03_modeling_kmeans.py first.")

# Load data with cluster labels
data = pd.read_csv(in_path)
labels = data['cluster']
X = data.drop(columns=['cluster'])

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# Plot and save
os.makedirs("results/figures", exist_ok=True)
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', s=10)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Spotify Clusters (PCA)')
plt.legend(*scatter.legend_elements(), title="cluster")
plt.savefig(out_path_fig, dpi=150)
plt.close()  # close figure to prevent blocking

# Print cluster sizes
print("Clusters PCA visualization done.")
print("Figure saved to:", out_path_fig)
print("Cluster sizes:")
print(data['cluster'].value_counts().sort_index())
