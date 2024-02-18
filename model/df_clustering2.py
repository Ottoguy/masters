import os
import glob
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from joblib import Parallel, delayed
import time

# Record start time
start_time = time.time()

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

# Record time after loading data
load_data_time = time.time() - start_time
print(f"Time taken to load data: {load_data_time:.2f} seconds")

# Extract relevant columns
time_series_data = df[['ID', 'VoltageDiff', 'CurrentDiff']].copy()

# Reshape the DataFrame with variable length time series
time_series_list = []

for id_value, group in time_series_data.groupby('ID'):
    features = group[['VoltageDiff', 'CurrentDiff']].values
    time_series_list.append(features)

# Convert to time series dataset if needed
time_series_dataset = to_time_series_dataset(time_series_list)

# Print the shape of the reshaped data
print("Reshaped Data Shape:", time_series_dataset.shape)

#Save the reshaped data to a file
reshaped_data_file_path = 'prints/reshaped_data.npy'
np.save(reshaped_data_file_path, time_series_dataset)

# Record time after reshaping data
reshape_data_time = time.time() - start_time
print(f"Time taken to reshape data: {reshape_data_time - load_data_time:.2f} seconds")

# Use joblib for parallel processing
num_cores = -1  # Set to -1 to use all available cores

scaled_data_file_path = 'prints/scaled_data.npy'

def scaling_data(time_series_dataset):
    # Scale the time series data
    print("Scaling time series data...")
    scaler = TimeSeriesScalerMeanVariance()
    scaled_data = Parallel(n_jobs=num_cores)(
        delayed(scaler.fit_transform)(ts) for ts in time_series_dataset
    )
    # Save the scaled data to a file
    np.save(scaled_data_file_path, scaled_data)
    return scaler.fit_transform(time_series_dataset)

# Check if scaled data file exists
if os.path.exists(scaled_data_file_path):
    # Ask the user if they want to use the existing scaled data
    user_input = input("Scaled data already exists. Do you want to use it? (y/n): ").lower()
    if user_input == 'y':
        # Load the existing scaled data
        scaled_data = np.load(scaled_data_file_path)
    else:
        scaled_data = scaling_data(time_series_dataset)
else:
    scaled_data = scaling_data(time_series_dataset)

# Record time after scaling data
scale_data_time = time.time() - start_time
print(f"Time taken to scale data: {scale_data_time - reshape_data_time:.2f} seconds")

# Choose the number of clusters (you may need to experiment with this)
num_clusters = 3

# Apply TimeSeriesKMeans clustering with DTW as the metric
print("Clustering time series data...")
kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", n_jobs=num_cores, verbose=True)
print("Fitting the model...")
labels = kmeans.fit_predict(scaled_data)

# Record time after clustering
cluster_data_time = time.time() - start_time
print(f"Time taken to cluster data: {cluster_data_time - scale_data_time:.2f} seconds")

# Save the clustered data to a file
clustered_data_file_path = 'clustered_data.csv'
clustered_data = pd.DataFrame({'ID': time_series_data['ID'].unique(), 'Cluster': labels})
clustered_data.to_csv(clustered_data_file_path, index=False)

#Calculate silhouette score
silhouette_score(scaled_data, labels, metric="dtw", n_jobs=num_cores)

# Record time after saving clustered data
save_clustered_data_time = time.time() - start_time
print(f"Time taken to save clustered data: {save_clustered_data_time - cluster_data_time:.2f} seconds")

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