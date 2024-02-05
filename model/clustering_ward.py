import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from datetime import datetime
import mplcursors  # Import mplcursors

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

# Exclude the "ID" column from numeric features
data_numeric = data.select_dtypes(include='number').drop(columns=['ID'])

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

# Create a subplot with two columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Visualize the hierarchical clustering tree
dendrogram(linkage_matrix, labels=data.index, leaf_rotation=90, leaf_font_size=8, ax=ax1)
ax1.set_title('Ward\'s Hierarchical Clustering Dendrogram')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Distance')

for cluster_num in set(clusters):
    cluster_data = data_pca[clusters == cluster_num]
    ids = data['ID'][clusters == cluster_num]
    scatter = ax2.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
        label=f'Cluster {cluster_num}'
    )

mplcursors.cursor(hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(data['ID'].iloc[sel.target.index])
)

# Add legend to the scatter plot
ax2.legend()

# Access loadings (components)
loadings = pca.components_

# Print loadings with annotations
print("Loadings:")
for i, component in enumerate(loadings):
    print(f"PCA Component {i + 1}:")
    for j, loading in enumerate(component):
        print(f"  Feature {data_numeric.columns[j]}: {loading}")

# Save the figure with the current date and time in the filename
results_dir = "plots/clustering/ward/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
plt.show()
