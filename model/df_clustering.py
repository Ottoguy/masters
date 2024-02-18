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

# Reshape the DataFrame with variable length time series
time_series_list = []

for id_value, group in time_series_data.groupby('ID'):
    features = group[['VoltageDiff', 'CurrentDiff']].values
    time_series_list.append(features)

# Convert to time series dataset if needed
time_series_dataset = to_time_series_dataset(time_series_list)

# Print the shape of the reshaped data
print("Reshaped Data Shape:", time_series_dataset.shape)

# Check if scaled data file exists for VoltageDiff
scaled_data_file_path = 'prints/scaled_data.npy'
if os.path.exists(scaled_data_file_path):
    # Ask the user if they want to use the existing scaled data
    user_input = input("Scaled data already exists. Do you want to use it? (y/n): ").lower()
    if user_input == 'y':
        # Load the existing scaled data
        scaled_data = np.load(scaled_data_file_path)
    else:
        # Scale the time series data
        print("Scaling time series data...")
        scaler = TimeSeriesScalerMeanVariance()
        scaled_data = scaler.fit_transform(time_series_dataset)
        # Save the scaled data to a file
        np.save(scaled_data_file_path, scaled_data)
else:
    # Scale the time series data
    print("Scaling time series data...")
    scaler = TimeSeriesScalerMeanVariance()
    scaled_data = scaler.fit_transform(time_series_dataset)
    # Save the scaled data to a file
    np.save(scaled_data_file_path, scaled_data)

# Choose the number of clusters (you may need to experiment with this)
num_clusters = 3

# Apply TimeSeriesKMeans clustering with DTW as the metric
print("Clustering time series data...")
kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", verbose=True)
labels = kmeans.fit_predict(scaled_data)

# Save the clustered data to a file
clustered_data_file_path = 'clustered_data.csv'
clustered_data = pd.DataFrame({'ID': time_series_data['ID'].unique(), 'Cluster': labels})
clustered_data.to_csv(clustered_data_file_path, index=False)

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