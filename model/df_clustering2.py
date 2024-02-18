import os
import glob
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from joblib import Parallel, delayed
from functions import export_csv_for_id

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

# Select only the first hundredth of the data
time_series_data_subset = time_series_data.head(len(time_series_data) // 20)

# Reshape the DataFrame with variable length time series
time_series_list = []

# USE ENTIRE DATA OR SUBSET HERE
for id_value, group in time_series_data_subset.groupby('ID'):
    # Separate VoltageDiff and CurrentDiff
    voltage_diff = group['VoltageDiff'].values
    current_diff = group['CurrentDiff'].values
    
    # Scale each feature separately
    scaler_voltage = TimeSeriesScalerMeanVariance()
    scaled_voltage_diff = scaler_voltage.fit_transform(voltage_diff.reshape(-1, 1))
    
    scaler_current = TimeSeriesScalerMeanVariance()
    scaled_current_diff = scaler_current.fit_transform(current_diff.reshape(-1, 1))
    
    # Combine the scaled features into one array
    scaled_features = np.hstack((scaled_voltage_diff, scaled_current_diff))
    
    # Reshape to (length, dimensions) to avoid the broadcasting issue
    scaled_features = scaled_features.reshape(len(scaled_features), -1)
    
    time_series_list.append(scaled_features)

# Convert to time series dataset if needed
time_series_dataset = to_time_series_dataset(time_series_list)
#Save the reshaped data to a file
reshaped_data_file_path = 'prints/reshaped_data.npy'
np.save(reshaped_data_file_path, time_series_dataset)

# Use joblib for parallel processing
num_cores = -1  # Set to -1 to use all available cores

export_csv_for_id(scaled_data, "dtw")

# Choose the number of clusters (you may need to experiment with this)
num_clusters = 3
warping_window = 10  # Set your desired warping window size

# Apply TimeSeriesKMeans clustering with DTW as the metric
print("Clustering time series data...")
kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", n_jobs=num_cores, verbose=True, init='random')
labels = kmeans.fit_predict(scaled_data)

print("Labels shape:", labels.shape)
print("Labels:", labels)
print("time_series_data['ID'].unique() shape:", time_series_data['ID'].unique().shape)

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