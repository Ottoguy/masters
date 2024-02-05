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

# Iterate over 10 plausible values for n_components
for n_components in range(2, 12):
    # Use Gaussian Mixture Model for clustering
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    clusters = gmm.fit_predict(data_pca)

    # Create subplots
    fig, ax = plt.subplots(figsize=(8, 6))

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

    # Add labels and title to the scatter plot
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title(f'Gaussian Mixture Model Clustering (n_components={n_components})')

    # Add legend to the scatter plot
    legend_labels = [f'Cluster {i}' for i in range(n_components)]
    ax.legend(legend_labels, loc='upper right')

    # Add means and covariances information to the scatter plot
    means = gmm.means_
    covariances = gmm.covariances_
    for i, (mean, covariance) in enumerate(zip(means, covariances)):
        info_text = f"Cluster {i}:\nMean: {mean}\nCovariance: {covariance}"
        ax.text(0.05, 0.9 - i * 0.1, info_text, transform=ax.transAxes, fontsize=8, verticalalignment='top')

    # Save the figure with the current date and time in the filename
    results_dir = "plots/clustering/gmm/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(results_dir, f'n_components_{n_components}_{current_datetime}.png'))
    plt.show()
