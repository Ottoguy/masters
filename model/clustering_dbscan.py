import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors
from datetime import datetime
import numpy as np

# Specify the directory where your files are located
folder_path = 'prints/meta/'

# Create a pattern to match files in the specified format
file_pattern = '*'

# Get a list of all files matching the pattern
file_list = glob.glob(os.path.join(folder_path, file_pattern))

# Sort the files based on modification time (latest first)
file_list.sort(key=os.path.getmtime, reverse=True)

# Take the latest file
latest_file = file_list[0]

# Load your data from the latest file
data = pd.read_csv(latest_file)

# Assuming you want to cluster based on certain features, drop non-numeric columns if needed
data_numeric = data.select_dtypes(include='number').drop(columns=['ID'])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Perform PCA for dimensionality reduction (optional but can be helpful)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Generate 10 plausible eps values between 0.05 and 0.5
eps_values = np.linspace(0.1, 0.35, 10)

#Round the eps values to 2 decimal places
eps_values = np.round(eps_values, 2)

# Define discernible colors for clusters
cluster_colors = [
    'red', 'green', 'blue', 'purple', 'orange',
    'cyan', 'magenta', 'yellow', 'brown', 'pink', 'olive', 'skyblue', 'lightgreen', 'lightblue', 'lightcoral', 'lightpink', 'lightyellow', 'lightgrey', 'lightcyan', 'lightseagreen', 'lightsalmon', 'lightgreen', 'lightblue', 'lightcoral', 'lightpink', 'lightyellow', 'lightgrey', 'lightcyan', 'lightseagreen', 'lightsalmon'
]

# Create subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.flatten()

# Iterate over eps values
for i, eps in enumerate(eps_values):
    # Use DBSCAN for clustering
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(data_pca)

    # Visualize the clusters in 2D space using Matplotlib
    ax = axes[i]
    ax.set_title(f'DBSCAN Clustering (eps={eps})')

    unique_clusters = set(clusters)
    num_colors_needed = max(len(unique_clusters), len(cluster_colors))

    for j, cluster_num in enumerate(unique_clusters):
        if cluster_num != -1:  # Exclude outliers
            cluster_data = data_pca[clusters == cluster_num]
            color_index = j % len(cluster_colors)
            ax.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                c=[cluster_colors[color_index]] * len(cluster_data),
                label=f'Cluster {cluster_num}'
            )

    ax.legend()
    
    # Set the same x and y-axis limits for all subplots
    ax.set_xlim(-3.5, 3)
    ax.set_ylim(-2.5, 2)

# Access loadings (components)
loadings = pca.components_

# Print loadings with annotations
print("Loadings:")
for i, component in enumerate(loadings):
    print(f"PCA Component {i + 1}:")
    for j, loading in enumerate(component):
        print(f"  Feature {data_numeric.columns[j]}: {loading}")

mplcursors.cursor(hover=True)  # Enable mplcursors for hover text

# Adjust layout
plt.tight_layout()

# Save the figure with the current date and time in the filename
results_dir = "plots/clustering/dbscan/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
plt.show()