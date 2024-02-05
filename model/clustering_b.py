import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors
from datetime import datetime

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
data_numeric = data.select_dtypes(include='number')

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Perform PCA for dimensionality reduction (optional but can be helpful)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Use DBSCAN for clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
clusters = dbscan.fit_predict(data_pca)

# Visualize the clusters in 2D space using Matplotlib and mplcursors
fig, ax = plt.subplots()

for cluster_num in set(clusters):
    if cluster_num == -1:  # Outliers are labeled as -1 by DBSCAN
        cluster_data = data_pca[clusters == cluster_num]
        ids = data['ID'][clusters == cluster_num]
        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            color='black',
            label=f'Outliers'
        )
    else:
        cluster_data = data_pca[clusters == cluster_num]
        ids = data['ID'][clusters == cluster_num]
        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            c=[cluster_num] * len(cluster_data),
            label=f'Cluster {cluster_num}'
        )

mplcursors.cursor(hover=True)  # Enable mplcursors for hover text

# Add labels and title
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title('DBSCAN Outlier Identification')

# Save the figure with the current date and time in the filename
results_dir = "plots/clustering_b/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
plt.show()