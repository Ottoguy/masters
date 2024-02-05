import os
import glob
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mplcursors import cursor
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

# Create subplots for different numbers of clusters (2 to 7) in two rows with three subplots each
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.flatten()

# Iterate over different numbers of clusters (2 to 7)
for n_clusters, ax in zip(range(2, 8), axes):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    
    # Create a scatter plot for the current number of clusters
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    
    # Add hover text for IDs using mplcursors
    cursor(hover=True).connect("add", lambda sel, ax=ax: sel.annotation.set_text(f"ID: {data['ID'][sel.target.index]}"))

    # Set labels and title
    ax.set_title(f'Clusters: {n_clusters}')

# Set labels and title for the entire figure
fig.suptitle('K-Means Clustering with Varying Number of Clusters (2 to 7)', y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.96])

# Add a legend (you may need to adjust this based on your specific data)
legend_labels = [f'Cluster {cluster_num}' for cluster_num in range(n_clusters)]
fig.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0))

# Access loadings (components)
loadings = pca.components_

# Print loadings with annotations
print("Loadings:")
for i, component in enumerate(loadings):
    print(f"PCA Component {i + 1}:")
    for j, loading in enumerate(component):
        print(f"  Feature {data_numeric.columns[j]}: {loading}")

# Save the figure with the current date and time in the filename
results_dir = "plots/clustering/kmeans/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
plt.show()