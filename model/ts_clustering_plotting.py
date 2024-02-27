import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from functions import export_csv_for_id
from datetime import datetime

# Load the data from the latest CSV file containing time series data
ts_files = glob.glob('prints/extracted/*.csv')
latest_ts_file = max(ts_files, key=os.path.getmtime)
df_ts = pd.read_csv(latest_ts_file)

origin = "3-Phase_24_clusters"
# Specify the directory where your ID-cluster mapping files are located
id_cluster_folder_path = 'prints/ts_clustering/' + origin + '/'

# Get a list of all files in the specified format within the chosen subfolder
id_cluster_files = glob.glob(os.path.join(id_cluster_folder_path, '*.csv'))

# Sort the files based on modification time (latest first)
id_cluster_files.sort(key=os.path.getmtime, reverse=True)

# Take the latest file from the chosen subfolder
latest_id_cluster_file = id_cluster_files[0]

# Load your ID-cluster mapping data from the latest file
df_clusters = pd.read_csv(latest_id_cluster_file)

# Add the cluster feature to the time series data
df_ts['Cluster'] = df_ts['ID'].map(df_clusters.set_index('ID')['Cluster'])

# Sort by ID and timestamp
df_ts = df_ts.sort_values(by=['ID', 'Timestamp'])

# Calculate the number of clusters
num_clusters = df_clusters['Cluster'].nunique()
print(f"Number of clusters: {num_clusters}")

current_columns = ['Phase1Current', 'Phase2Current', 'Phase3Current']
voltage_columns = ['Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']

# Create subplots with one row per cluster and two columns (current and voltage)
fig, axes = plt.subplots(num_clusters, 2, figsize=(12, 4*num_clusters), sharex=True)

# Iterate through each cluster
for cluster_id in range(0, num_clusters):
    # Select data for the current cluster
    cluster_data = df_ts[df_ts['Cluster'] == cluster_id]

    # Iterate through each ID in the cluster
    for i, (_, id_data) in enumerate(cluster_data.groupby('ID')):
        # Select the relevant columns for the plot
        x_values = range(1, 61)  # 60 ticks as mentioned

        # Plot all three phases of current in the left column
        for phase_idx, current_column in enumerate(current_columns):
            current_values = id_data[current_column].values

            if len(current_values) != 60:
                print(f"ID {id_data['ID'].iloc[0]} does not have 60 ticks")
                continue

            axes[cluster_id, 0].plot(x_values, current_values, label=f'{current_column} - ID {id_data["ID"].iloc[0]}')

        # Plot all three phases of voltage in the right column
        for phase_idx, voltage_column in enumerate(voltage_columns):
            voltage_values = id_data[voltage_column].values

            if len(voltage_values) != 60:
                print(f"ID {id_data['ID'].iloc[0]} does not have 60 ticks")
                continue

            axes[cluster_id, 1].plot(x_values, voltage_values, label=f'{voltage_column} - ID {id_data["ID"].iloc[0]}')

    # Set titles, labels, legends, etc. for each subplot
    axes[cluster_id, 0].set_title(f'Cluster {cluster_id} - Current')
    axes[cluster_id, 0].set_ylabel('Current')
    axes[cluster_id, 0].grid(True)
    #axes[cluster_id, 0].legend()

    axes[cluster_id, 1].set_title(f'Cluster {cluster_id} - Voltage')
    axes[cluster_id, 1].set_ylabel('Voltage')
    axes[cluster_id, 1].grid(True)
    #axes[cluster_id, 1].legend()

# Add overall titles, labels, etc. for the entire figure
plt.suptitle(f'{origin} – Current and Voltage Time Series for {num_clusters} clusters')
plt.xlabel('Time')
plt.tight_layout()

results_dir = "plots/ts_clustering/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save the figure with the current date and time in the filename
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Saving figure {current_datetime}", end='\r')
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
plt.close()
