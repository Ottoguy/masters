import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

# Visualize the clusters in 2D space using Plotly
fig = go.Figure()

for cluster_num in set(clusters):
    if cluster_num == -1:  # Outliers are labeled as -1 by DBSCAN
        cluster_data = data_pca[clusters == cluster_num]
        ids = data['ID'][clusters == cluster_num]
        fig.add_trace(go.Scatter(
            x=cluster_data[:, 0],
            y=cluster_data[:, 1],
            mode='markers',
            hovertext=ids,
            marker=dict(color='black'),
            name=f'Outliers'
        ))
    else:
        cluster_data = data_pca[clusters == cluster_num]
        ids = data['ID'][clusters == cluster_num]
        fig.add_trace(go.Scatter(
            x=cluster_data[:, 0],
            y=cluster_data[:, 1],
            mode='markers',
            hovertext=ids,
            marker=dict(color=cluster_num),
            name=f'Cluster {cluster_num}'
        ))

fig.update_layout(title='DBSCAN Clustering with Hover Text')
fig.show()
