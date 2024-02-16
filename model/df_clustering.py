import os
import glob
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series
import matplotlib.pyplot as plt

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
time_series_data = df[['ID', 'VoltageDiff', 'CurrentDiff']]

print(time_series_data.head())

# Handle missing values if any and create a copy
time_series_data = time_series_data.dropna().copy()

# Extract the time series data for clustering
time_series_voltage = time_series_data.pivot(index='ID', columns='VoltageDiff', values='VoltageDiff').values
time_series_current = time_series_data.pivot(index='ID', columns='CurrentDiff', values='CurrentDiff').values

# Convert the data to time series format
X_voltage = to_time_series(time_series_voltage)
X_current = to_time_series(time_series_current)

# Check if scaled data file exists for VoltageDiff
scaled_data_voltage_file_path = 'scaled_data_voltage.npy'
if os.path.exists(scaled_data_voltage_file_path):
    # Ask the user if they want to use the existing scaled data
    user_input = input("Scaled data for VoltageDiff already exists. Do you want to use it? (y/n): ").lower()
    if user_input == 'y':
        # Load the existing scaled data
        scaled_data_voltage = np.load(scaled_data_voltage_file_path)
    else:
        # Scale the time series data
        print("Scaling time series data for VoltageDiff...")
        scaler = TimeSeriesScalerMeanVariance()
        scaled_data_voltage = scaler.fit_transform(X_voltage)
        # Save the scaled data to a file
        np.save(scaled_data_voltage_file_path, scaled_data_voltage)
else:
    # Scale the time series data
    print("Scaling time series data for VoltageDiff...")
    scaler = TimeSeriesScalerMeanVariance()
    scaled_data_voltage = scaler.fit_transform(X_voltage)
    # Save the scaled data to a file
    np.save(scaled_data_voltage_file_path, scaled_data_voltage)

# Check if scaled data file exists for CurrentDiff
scaled_data_current_file_path = 'scaled_data_current.npy'
if os.path.exists(scaled_data_current_file_path):
    # Ask the user if they want to use the existing scaled data
    user_input = input("Scaled data for CurrentDiff already exists. Do you want to use it? (y/n): ").lower()
    if user_input == 'y':
        # Load the existing scaled data
        scaled_data_current = np.load(scaled_data_current_file_path)
    else:
        # Scale the time series data
        print("Scaling time series data for CurrentDiff...")
        scaler = TimeSeriesScalerMeanVariance()
        scaled_data_current = scaler.fit_transform(X_current)
        # Save the scaled data to a file
        np.save(scaled_data_current_file_path, scaled_data_current)
else:
    # Scale the time series data
    print("Scaling time series data for CurrentDiff...")
    scaler = TimeSeriesScalerMeanVariance()
    scaled_data_current = scaler.fit_transform(X_current)
    # Save the scaled data to a file
    np.save(scaled_data_current_file_path, scaled_data_current)

# Choose the number of clusters (you may need to experiment with this)
num_clusters = 3

# Apply TimeSeriesKMeans clustering with DTW as the metric for VoltageDiff
print("Clustering time series data for VoltageDiff...")
kmeans_voltage = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", verbose=True)
labels_voltage = kmeans_voltage.fit_predict(scaled_data_voltage)

# Apply TimeSeriesKMeans clustering with DTW as the metric for CurrentDiff
print("Clustering time series data for CurrentDiff...")
kmeans_current = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", verbose=True)
labels_current = kmeans_current.fit_predict(scaled_data_current)

# Save the clustered data to a file for VoltageDiff
clustered_data_voltage_file_path = 'clustered_data_voltage.csv'
clustered_data_voltage = pd.DataFrame({'ID': time_series_data['ID'].unique(), 'Cluster': labels_voltage})
clustered_data_voltage.to_csv(clustered_data_voltage_file_path, index=False)

# Save the clustered data to a file for CurrentDiff
clustered_data_current_file_path = 'clustered_data_current.csv'
clustered_data_current = pd.DataFrame({'ID': time_series_data['ID'].unique(), 'Cluster': labels_current})
clustered_data_current.to_csv(clustered_data_current_file_path, index=False)

# Visualize the clusters for VoltageDiff
print("Visualizing clusters for VoltageDiff...")
for cluster_id in range(num_clusters):
    cluster_subset = clustered_data_voltage[clustered_data_voltage['Cluster'] == cluster_id]
    plt.scatter(cluster_subset['ID'], [cluster_id] * len(cluster_subset), label=f'Cluster {cluster_id}')

plt.xlabel('ID')
plt.ylabel('Cluster')
plt.title('Time Series Clustering with DTW for VoltageDiff')
plt.legend()
plt.show()

# Visualize the clusters for CurrentDiff
print("Visualizing clusters for CurrentDiff...")
for cluster_id in range(num_clusters):
    cluster_subset = clustered_data_current[clustered_data_current['Cluster'] == cluster_id]
    plt.scatter(cluster_subset['ID'], [cluster_id] * len(cluster_subset), label=f'Cluster {cluster_id}')

plt.xlabel('ID')
plt.ylabel('Cluster')
plt.title('Time Series Clustering with DTW for CurrentDiff')
plt.legend()
plt.show()
