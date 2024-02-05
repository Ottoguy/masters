import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
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

# Use Gaussian Mixture Model for clustering
n_components = 3  # Adjust as needed
gmm = GaussianMixture(n_components=n_components, random_state=42)
clusters = gmm.fit_predict(data_pca)

# Create a subplot with two columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Visualize the clusters in 2D space
for cluster_num in range(n_components):
    cluster_data = data_pca[clusters == cluster_num]
    ids = data['ID'][clusters == cluster_num]
    ax2.scatter(
        cluster_data[:, 0],
        cluster_data[:, 1],
        label=f'Cluster {cluster_num}'
    )

# Add labels and title to the scatter plot
ax2.set_xlabel('PCA Component 1')
ax2.set_ylabel('PCA Component 2')
ax2.set_title('Gaussian Mixture Model Clustering')

# Add legend to the scatter plot
ax2.legend()

# Access means and covariances of the components
means = gmm.means_
covariances = gmm.covariances_

# Print means and covariances
print("Means:")
print(means)
print("\nCovariances:")
print(covariances)

# Save the figure with the current date and time in the filename
results_dir = "plots/clustering/gmm/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
plt.show()
