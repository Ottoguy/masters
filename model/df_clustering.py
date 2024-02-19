import os
import glob
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from functions import export_csv_for_id
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Use joblib for parallel processing
num_cores = -1  # Set to -1 to use all available cores

# Specify the directory where your files are located
folder_path = 'prints/extracted/'

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

# Reshape the DataFrame with variable length time series
time_series_list = []

for id_value, group in time_series_data.groupby('ID'):
    features = group[['Phase1Voltage', 'Phase1Current']].values
    time_series_list.append(features)

# Convert to time series dataset if needed
time_series_dataset = to_time_series_dataset(time_series_list)

# Print the shape of the reshaped data
print("Reshaped Data Shape:", time_series_dataset.shape)

scaled_data=time_series_dataset

# Choose the number of clusters (you may need to experiment with this)
num_clusters = 6

# Apply TimeSeriesKMeans clustering with DTW as the metric
print("Clustering time series data...")
kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", n_jobs=num_cores, verbose=True)
labels = kmeans.fit_predict(scaled_data)

#Calculate silhouette score
s_score = silhouette_score(scaled_data, labels, metric="dtw", n_jobs=num_cores)

print("Saving clustered data to a file...")
# Save the figure with the current date and time in the filename
results_dir = "prints/ts_clustering/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the clustered data to a file
clustered_data_file_path = 'clustered_data_'
clustered_data = pd.DataFrame({'ID': time_series_data['ID'].unique(), 'Cluster': labels, 'Silhouette Score': s_score, 'Num Clusters': num_clusters})
clustered_data.to_csv(results_dir + clustered_data_file_path + current_datetime + ".csv", index=False)

#Print stats for each cluster
print("Printing stats for each cluster...")
for i in range(num_clusters):
    print(f"Cluster {i} stats:")
    print(clustered_data[clustered_data['Cluster'] == i].describe())
