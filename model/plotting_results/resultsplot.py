import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def resultsplot():# Specify the directory where your files are located
    folder_path = 'prints/dl_merge/'

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

    # Sort by lowest RMSE
    data = data.sort_values(by=['RMSE_Clusters'])

    # Save the first row for every unique combination of "TS_Samples" and "Clusters" and delete the rest
    data = data.drop_duplicates(subset=['TS_Samples', 'Clusters'])

    # Group data by 'TS_Samples'
    grouped_data = data.groupby('TS_Samples')

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
    data = data.sort_values(by=['RMSE_Intermediate'])
    
    # Group data by 'TS_Samples'
    grouped_data = data.groupby('TS_Samples')

    # Plot each group separately
    for i, (name, group) in enumerate(grouped_data):
        # Sort the data within the group based on 'Clusters' column
        sorted_group = group.sort_values(by='Clusters')
        # Plot scatter points and lines for "RMSE_Intermediate"
        plt.scatter(sorted_group['Clusters'], sorted_group['RMSE_Intermediate'], label=f"TS_Samples = {name} (Intermediate)", color=colors[i], marker='x')
        plt.plot(sorted_group['Clusters'], sorted_group['RMSE_Intermediate'], linestyle='--', color=colors[i])

    plt.xlabel('Clusters')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Clusters for different TS_Samples')
    plt.xticks(np.arange(2, 21, step=1))  # Set x-axis ticks from 2 to 20
    plt.legend()
    plt.grid(True)
    plt.show()

resultsplot()
