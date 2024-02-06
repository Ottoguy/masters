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

# Set up five plausible threshold values
threshold_values = [6, 8, 10, 12, 14]

# Create a subplot grid for the six plots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Hierarchical Clustering Results for Different Thresholds')

# Use Ward's hierarchical clustering for dendrogram
linkage_matrix_dendrogram = linkage(data_pca, method='ward')
clusters_dendrogram = fcluster(linkage_matrix_dendrogram, 20, criterion='distance')

# Plot dendrogram in the first subplot
ax_dendrogram = axs[-1, -1]
dendrogram(linkage_matrix_dendrogram, labels=data.index, leaf_rotation=90, leaf_font_size=8, ax=ax_dendrogram, above_threshold_color='y', color_threshold=20)
ax_dendrogram.set_title('Dendrogram')
ax_dendrogram.set_xlabel('Sample Index')
ax_dendrogram.set_ylabel('Distance')

# Iterate over five threshold values and plot scatter plots
for i, threshold in enumerate(threshold_values):
    # Use Ward's hierarchical clustering
    linkage_matrix = linkage(data_pca, method='ward')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    # Create a subplot for scatter plot
    ax_scatter = axs[i // 3, i % 3]

    # Plot scatter plot for each cluster
    for cluster_num in set(clusters):
        cluster_data = data_pca[clusters == cluster_num]
        ids = data['ID'][clusters == cluster_num]
        scatter = ax_scatter.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            label=f'Cluster {cluster_num}'
        )

    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(data['ID'].iloc[sel.target.index])
    )

    ax_scatter.set_title(f'Scatter Plot (Threshold: {threshold})')
    ax_scatter.legend(loc='upper left')

# Access loadings (components)
loadings = pca.components_

# Print loadings with annotations
print("Loadings:")
for i, component in enumerate(loadings):
    print(f"PCA Component {i + 1}:")
    for j, loading in enumerate(component):
        print(f"  Feature {data_numeric.columns[j]}: {loading}")

# Adjust layout and show the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
results_dir = "plots/clustering/gmm/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, f"all_n_components_{current_datetime}.png"))
plt.show()
