import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def resultsplot():# Specify the directory where your files are located
    folder_path = 'prints/backup_oldruns/dl_merge/'

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

    # Delete all duplicate rows
    data = data.drop_duplicates()

    # Delete all columns except "TS_Samples", "Clusters" and "RMSE_Clusters"
    data = data[['TS_Samples', 'Clusters', 'RMSE_Clusters', 'RMSE_Intermediate']]

    # Delete rows with TS_Samples = 10
    data = data[data.TS_Samples != 10]

    #Remove rows with NaN values in 'TS_Samples' column
    data = data.dropna(subset=['TS_Samples'])

    # Sort by lowest RMSE
    data = data.sort_values(by=['RMSE_Clusters'])

    # Sort by lowest RMSE_Intermediate within each unique combination of Clusters and TS_Samples
    data = data.sort_values(by=['TS_Samples', 'Clusters', 'RMSE_Intermediate'])

    # Save the first row for every unique combination of Clusters and TS_Samples based on RMSE_Intermediate
    best_intermediate = data.drop_duplicates(subset=['TS_Samples', 'Clusters'])

    # Sort by lowest RMSE_Clusters within each unique combination of Clusters and TS_Samples
    data = data.sort_values(by=['TS_Samples', 'Clusters', 'RMSE_Clusters'])

    # Save the first row for every unique combination of Clusters and TS_Samples based on RMSE_Clusters
    best_clusters = data.drop_duplicates(subset=['TS_Samples', 'Clusters'])

    # Merge the best_intermediate and best_clusters dataframes on 'TS_Samples' and 'Clusters'
    merged_data = best_intermediate.merge(best_clusters, on=['TS_Samples', 'Clusters'], suffixes=('_intermediate', '_clusters'))

    #Delete the columns 'RMSE_Intermediate_clusters' and 'RMSE_Clusters_intermediate' from the merged dataframe
    merged_data = merged_data.drop(columns=['RMSE_Intermediate_clusters', 'RMSE_Clusters_intermediate'])

    #Rename the columns RMSE_Intermediate_intermediate and RMSE_Clusters_clusters to RMSE_Intermediate and RMSE_Clusters
    merged_data = merged_data.rename(columns={'RMSE_Intermediate_intermediate': 'RMSE_Intermediate', 'RMSE_Clusters_clusters': 'RMSE_Clusters'})

    # Group data by 'TS_Samples'
    grouped_data = merged_data.groupby('TS_Samples')

    # Define colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_data)))

    # Plot each group separately
    for i, (name, group) in enumerate(grouped_data):
        # Sort the data within the group based on 'Clusters' column
        sorted_group = group.sort_values(by='Clusters')
        # Plot scatter points and lines for "RMSE_Clusters"
        plt.scatter(sorted_group['Clusters'], sorted_group['RMSE_Clusters'], label=f"TS_Samples = {name}", color=colors[i])
        plt.plot(sorted_group['Clusters'], sorted_group['RMSE_Clusters'], linestyle='-', color=colors[i])

    # Sort by lowest RMSE
    merged_data = merged_data.sort_values(by=['RMSE_Intermediate'])
    
    # Group data by 'TS_Samples'
    grouped_data = merged_data.groupby('TS_Samples')

    # Plot each group separately
    for i, (name, group) in enumerate(grouped_data):
        # Sort the data within the group based on 'Clusters' column
        sorted_group = group.sort_values(by='Clusters')
        # Plot scatter points and lines for "RMSE_Intermediate"
        plt.scatter(sorted_group['Clusters'], sorted_group['RMSE_Intermediate'], label=f"TS_Samples = {name} (Intermediate)", color=colors[i], marker='x')
        plt.plot(sorted_group['Clusters'], sorted_group['RMSE_Intermediate'], linestyle='--', color=colors[i])

    plt.xlabel('Clusters', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.title('RMSE vs Clusters for different TS_Samples', fontsize=16)
    plt.xticks(np.arange(2, 21, step=1))  # Set x-axis ticks from 2 to 20
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

resultsplot()
