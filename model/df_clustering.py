import os
import glob
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from functions import export_csv_for_id
from sklearn.preprocessing import StandardScaler

# Use joblib for parallel processing
num_cores = -1  # Set to -1 to use all available cores

# Specify the directory where your files are located
folder_path = 'prints/all/'

# Create a pattern to match files in the specified format
file_pattern = '*'

# Get a list of all files matching the pattern
file_list = glob.glob(os.path.join(folder_path, file_pattern))

# Sort the files based on modification time (latest first)
file_list.sort(key=os.path.getmtime, reverse=True)

# Take the latest file
latest_file = file_list[0]

# Load your data from the latest file
df = pd.read_csv(latest_file)

# Extract relevant columns
time_series_data = df[['ID', 'VoltageDiff', 'CurrentDiff']].copy()

print("Scaling time series data...")
# Separate standard scaling for 'VoltageDiff' and 'CurrentDiff'
scaler_voltage = StandardScaler()
scaler_current = StandardScaler()

# Fit and transform each feature
time_series_data['VoltageDiff'] = scaler_voltage.fit_transform(time_series_data[['VoltageDiff']].dropna())
time_series_data['CurrentDiff'] = scaler_current.fit_transform(time_series_data[['CurrentDiff']].dropna())

# Round the values to 4 decimals
time_series_data = time_series_data.round({'VoltageDiff': 4, 'CurrentDiff': 4})

# Reshape the DataFrame with variable length time series
time_series_list = []

for id_value, group in time_series_data.groupby('ID'):
    features = group[['VoltageDiff', 'CurrentDiff']].values
    time_series_list.append(features)

# Convert to time series dataset if needed
time_series_dataset = to_time_series_dataset(time_series_list)

# Print the shape of the reshaped data
print("Reshaped Data Shape:", time_series_dataset.shape)

#save the data to a npy file
reshaped_data_file_path = 'prints/reshaped_data.npy'
np.save(reshaped_data_file_path, time_series_dataset)

scaled_data=time_series_dataset

# Choose the number of clusters (you may need to experiment with this)
num_clusters = 12

# Apply TimeSeriesKMeans clustering with DTW as the metric
print("Clustering time series data...")
kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", n_jobs=num_cores, verbose=True)
labels = kmeans.fit_predict(scaled_data)

print("Saving clustered data to a file...")
# Save the clustered data to a file
clustered_data_file_path = 'clustered_data.csv'
clustered_data = pd.DataFrame({'ID': time_series_data['ID'].unique(), 'Cluster': labels})
clustered_data.to_csv(clustered_data_file_path, index=False)

#Calculate silhouette score
silhouette_score(scaled_data, labels, metric="dtw", n_jobs=num_cores)

#Save the silhouette score
silhouette_score_file_path = 'prints/silhouette_score.txt'
with open(silhouette_score_file_path, 'w') as f:
    f.write(f"Silhouette Score: {silhouette_score(scaled_data, labels, metric='dtw', n_jobs=num_cores)}")

# Visualize the clusters
print("Visualizing clusters...")
for cluster_id in range(num_clusters):
    cluster_subset = clustered_data[clustered_data['Cluster'] == cluster_id]
    plt.scatter(cluster_subset['ID'], [cluster_id] * len(cluster_subset), label=f'Cluster {cluster_id}')

plt.xlabel('ID')
plt.ylabel('Cluster')
plt.title('Time Series Clustering with DTW')
plt.legend()
plt.show()