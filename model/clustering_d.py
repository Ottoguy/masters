import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
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

# Use Ward's hierarchical clustering
linkage_matrix = linkage(data_pca, method='ward')

# Set the threshold for cutting the tree to get clusters
threshold = 10  # Adjust as needed
clusters = fcluster(linkage_matrix, threshold, criterion='distance')

# Visualize the hierarchical clustering tree
fig, ax = plt.subplots(figsize=(10, 6))
dendrogram(linkage_matrix, labels=data.index, leaf_rotation=90, leaf_font_size=8, ax=ax)
ax.set_title('Ward\'s Hierarchical Clustering Dendrogram')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Distance')

plt.show()

# Visualize the clusters in 2D space
fig, ax = plt.subplots()

for cluster_num in set(clusters):
    cluster_data = data_pca[clusters == cluster_num]
    ids = data['ID'][clusters == cluster_num]
    ax.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
        label=f'Cluster {cluster_num}'
    )

# Access loadings (components)
loadings = pca.components_

# Print loadings with annotations
print("Loadings:")
for i, component in enumerate(loadings):
    print(f"PCA Component {i + 1}:")
    for j, loading in enumerate(component):
        print(f"  Feature {data_numeric.columns[j]}: {loading}")

# Add labels and title
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title('Ward\'s Hierarchical Clustering')

plt.legend()

# Save the figure with the current date and time in the filename
results_dir = "plots/clustering_d/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
plt.show()
