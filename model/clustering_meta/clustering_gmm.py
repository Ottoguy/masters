import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
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

# Assuming you want to cluster based on certain features, drop non-numeric columns if needed
data_numeric = data.select_dtypes(include='number').drop(columns=['ID'])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Perform PCA for dimensionality reduction (optional but can be helpful)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Create a single figure with subplots for each value of n_components
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Gaussian Mixture Model Clustering', fontsize=16)

for n_components, ax in zip(range(2, 12), axs.flatten()):
    # Use Gaussian Mixture Model for clustering
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    clusters = gmm.fit_predict(data_pca)

    for cluster_num in set(clusters):
        cluster_data = data_pca[clusters == cluster_num]
        ids = data['ID'][clusters == cluster_num]
        scatter = ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            label=f'Cluster {cluster_num}'
        )

    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(data['ID'].iloc[sel.target.index])
    )

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title(f'n_components={n_components}')

    legend_labels = [f'Cluster {i}' for i in range(n_components)]
    ax.legend(legend_labels, loc='upper right')

# Print means and covariances for each cluster
for n_components in range(2, 12):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    clusters = gmm.fit_predict(data_pca)
    
    print(f"\nFor n_components={n_components}:")
    for i, (mean, covariance) in enumerate(zip(gmm.means_, gmm.covariances_)):
        print(f"Cluster {i} - Mean: {mean}, Covariance: {covariance}")

# Access loadings (components)
loadings = pca.components_

# Print loadings with annotations
print("Loadings:")
for i, component in enumerate(loadings):
    print(f"PCA Component {i + 1}:")
    for j, loading in enumerate(component):
        print(f"  Feature {data_numeric.columns[j]}: {loading}")

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
results_dir = "plots/clustering/gmm/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, f"{current_datetime}.png"))
plt.show()
