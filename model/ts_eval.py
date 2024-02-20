import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

meta_path = 'prints/meta/'
ts_path = 'prints/ts_clustering/'

file_pattern = '*'

meta_list = glob.glob(os.path.join(meta_path, file_pattern))
meta_list.sort(key=os.path.getmtime, reverse=True)
latest_meta = meta_list[0]
meta_df = pd.read_csv(latest_meta)

#Iterate over 1-Phase folders in ts_clustering
ts_clustering_1_phase_folders = glob.glob(os.path.join(ts_path, '1-Phase_*'))
#Make dataframes for the latest file in each folder
ts_clustering_1_phase_dfs = []
for folder in ts_clustering_1_phase_folders:
    id_cluster_files = glob.glob(os.path.join(folder, '*.csv'))
    id_cluster_files.sort(key=os.path.getmtime, reverse=True)
    latest_id_cluster_file = id_cluster_files[0]
    df_clusters = pd.read_csv(latest_id_cluster_file)
    ts_clustering_1_phase_dfs.append(df_clusters)

#Iterate over 3-Phase folders in ts_clustering
ts_clustering_3_phase_folders = glob.glob(os.path.join(ts_path, '3-Phase_*'))
#Make dataframes for the latest file in each folder
ts_clustering_3_phase_dfs = []
for folder in ts_clustering_3_phase_folders:
    id_cluster_files = glob.glob(os.path.join(folder, '*.csv'))
    id_cluster_files.sort(key=os.path.getmtime, reverse=True)
    latest_id_cluster_file = id_cluster_files[0]
    df_clusters = pd.read_csv(latest_id_cluster_file)
    ts_clustering_3_phase_dfs.append(df_clusters)

#Merge meta_df with each dataframe
for df in ts_clustering_1_phase_dfs:
    df = pd.merge(meta_df, df, on='ID')
for df in ts_clustering_3_phase_dfs:
    df = pd.merge(meta_df, df, on='ID')

# Function to calculate average values for each cluster
def calculate_cluster_averages(df):
    df['Average_Half_Minutes'] = df.groupby('Cluster')['Half_Minutes'].transform('mean')
    df['Average_Charging_Half_Minutes'] = df.groupby('Cluster')['Charging_Half_Minutes'].transform('mean')
    df.drop_duplicates(subset='Cluster', inplace=True)

# Calculate average values for each cluster in ts_clustering_1_phase_dfs
for df in ts_clustering_1_phase_dfs:
    calculate_cluster_averages(df)

# Calculate average values for each cluster in ts_clustering_3_phase_dfs
for df in ts_clustering_3_phase_dfs:
    calculate_cluster_averages(df)

def calculate_average_distance(dfs):
    # Combine dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)

    # Group by 'Num Clusters' and calculate the average distance between averages
    average_distance_df = combined_df.groupby('Num Clusters').agg({
        'Average_Half_Minutes': 'mean',
        'Average_Charging_Half_Minutes': 'mean'
    }).reset_index()

    # Calculate the average distance between averages
    average_distance_df['Average_Distance'] = np.sqrt(
        (average_distance_df['Average_Half_Minutes'] - average_distance_df['Average_Half_Minutes'].mean())**2 +
        (average_distance_df['Average_Charging_Half_Minutes'] - average_distance_df['Average_Charging_Half_Minutes'].mean())**2
    )

    return average_distance_df

# Calculate average distance for 1-Phase
average_distance_1_phase = calculate_average_distance(ts_clustering_1_phase_dfs)

# Calculate average distance for 3-Phase
average_distance_3_phase = calculate_average_distance(ts_clustering_3_phase_dfs)

print("Average Distance for 1-Phase:")
print(average_distance_1_phase)

print("\nAverage Distance for 3-Phase:")
print(average_distance_3_phase)

