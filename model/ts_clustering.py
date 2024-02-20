import os
import glob
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Use joblib for parallel processing
num_cores = -1  # Set to -1 to use all available cores

# Specify the range of clusters to iterate over
num_clusters_1_phase_range = range(3, 4)
num_clusters_3_phase_range = range(8, 9)

# Specify the directory where your files are located
folder_path = 'prints/extracted/'
meta_path = 'prints/meta/'

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

meta_list = glob.glob(os.path.join(meta_path, file_pattern))
meta_list.sort(key=os.path.getmtime, reverse=True)
latest_meta = meta_list[0]
meta_df = pd.read_csv(latest_meta)

# Extract relevant columns
time_series_data = df[['ID', 'Phase1Voltage', 'Phase1Current']].copy()

print("Scaling time series data...")
# Separate standard scaling for 'Phase1Voltage' and 'Phase1Current'
scaler_voltage = StandardScaler()
scaler_current = StandardScaler()

# Fit and transform each feature
time_series_data['Phase1Voltage'] = scaler_voltage.fit_transform(time_series_data[['Phase1Voltage']].dropna())
time_series_data['Phase1Current'] = scaler_current.fit_transform(time_series_data[['Phase1Current']].dropna())

# Round the values to 3 decimals
time_series_data = time_series_data.round({'Phase1Voltage': 3, 'Phase1Current': 3})

#Divide the time series data into one dataframe for Current_Type in meta_df = "1-Phase" and = "3-Phase"
time_series_data_1_phase = time_series_data[time_series_data['ID'].isin(meta_df[meta_df['Current_Type'] == "1-Phase"]['ID'])]
time_series_data_3_phase = time_series_data[time_series_data['ID'].isin(meta_df[meta_df['Current_Type'] == "3-Phase"]['ID'])]

# Reshape the DataFrame with variable length time series
time_series_1_phase_list = []
time_series_3_phase_list = []

for id_value, group in time_series_data_1_phase.groupby('ID'):
    features = group[['Phase1Voltage', 'Phase1Current']].values
    time_series_1_phase_list.append(features)

for id_value, group in time_series_data_3_phase.groupby('ID'):
    features = group[['Phase1Voltage', 'Phase1Current']].values
    time_series_3_phase_list.append(features)

# Convert to time series dataset if needed
time_series_1_phase_dataset = to_time_series_dataset(time_series_1_phase_list)
time_series_3_phase_dataset = to_time_series_dataset(time_series_3_phase_list)

# Print the shape of the reshaped data
print("Reshaped 1-Phase Data Shape:", time_series_1_phase_dataset.shape)
print("Reshaped 3-Phase Data Shape:", time_series_3_phase_dataset.shape)

# Iterate over different numbers of clusters
best_1_phase_s_score = float('-inf')
best_3_phase_s_score = float('-inf')
best_1_phase_num_clusters = None
best_3_phase_num_clusters = None

# Save the figure with the current date and time in the filename
results_dir = "prints/ts_clustering/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Iterate over different numbers of clusters
for num_clusters in num_clusters_1_phase_range:
    print(f"Clustering time series data with {num_clusters} clusters...")

    # Apply TimeSeriesKMeans clustering with DTW as the metric
    kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", n_jobs=num_cores, verbose=True)
    labels = kmeans.fit_predict(time_series_1_phase_dataset)

    # Calculate silhouette score
    s_score = silhouette_score(time_series_1_phase_dataset, labels, metric="dtw", n_jobs=num_cores)

    # Print silhouette score for the current number of clusters
    print(f"Silhouette Score for {num_clusters} clusters: {s_score}")

    # Create a subfolder for the current number of clusters
    subfolder_path = os.path.join(results_dir, f"1-Phase_{num_clusters}_clusters")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Save the clustered data to a file within the subfolder
    clustered_data_file_path = 'clustered_data_'
    clustered_data = pd.DataFrame(
        {'ID': time_series_data_1_phase['ID'].unique(), 'Cluster': labels, 'Silhouette Score': s_score, 'Num Clusters': num_clusters})
    clustered_data.to_csv(os.path.join(subfolder_path, clustered_data_file_path + current_datetime + ".csv"), index=False)

    # Print stats for each cluster within the subfolder
    print(f"Printing stats for each cluster in {num_clusters} clusters...")
    for i in range(num_clusters):
        cluster_stats = clustered_data[clustered_data['Cluster'] == i].describe()
        print(f"Cluster {i} stats:\n{cluster_stats}")

for num_clusters in num_clusters_3_phase_range:
    print(f"Clustering time series data with {num_clusters} clusters...")

    # Apply TimeSeriesKMeans clustering with DTW as the metric
    kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", n_jobs=num_cores, verbose=True)
    labels = kmeans.fit_predict(time_series_3_phase_dataset)

    # Calculate silhouette score
    s_score = silhouette_score(time_series_3_phase_dataset, labels, metric="dtw", n_jobs=num_cores)

    # Print silhouette score for the current number of clusters
    print(f"Silhouette Score for {num_clusters} clusters: {s_score}")

    # Create a subfolder for the current number of clusters
    subfolder_path = os.path.join(results_dir, f"3-Phase_{num_clusters}_clusters")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Save the clustered data to a file within the subfolder
    clustered_data_file_path = 'clustered_data_'
    clustered_data = pd.DataFrame(
        {'ID': time_series_data_3_phase['ID'].unique(), 'Cluster': labels, 'Silhouette Score': s_score, 'Num Clusters': num_clusters})
    clustered_data.to_csv(os.path.join(subfolder_path, clustered_data_file_path + current_datetime + ".csv"), index=False)

    # Print stats for each cluster within the subfolder
    print(f"Printing stats for each cluster in {num_clusters} clusters...")
    for i in range(num_clusters):
        cluster_stats = clustered_data[clustered_data['Cluster'] == i].describe()
        print(f"Cluster {i} stats:\n{cluster_stats}")

# Print the best number of clusters and its corresponding silhouette score
print(f"\nBest number of 1-Phase clusters: {best_1_phase_num_clusters}")
print(f"Silhouette Score for the best number of 1-Phaseclusters: {best_1_phase_s_score}")
print(f"\nBest number of 3-Phase clusters: {best_3_phase_num_clusters}")
print(f"Silhouette Score for the best number of 3-Phase clusters: {best_3_phase_s_score}")