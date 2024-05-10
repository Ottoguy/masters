import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def error_percentage():
    # Specify the directory where your files are located
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

    folder2_path = 'prints/backup_oldruns/dl_merge/'
    file_pattern2 = '*'
    file_list2 = glob.glob(os.path.join(folder2_path, file_pattern2))
    file_list2.sort(key=os.path.getmtime, reverse=True)
    latest_file2 = file_list2[0]
    data2 = pd.read_csv(latest_file2)

    #Merge the two dataframes
    data = pd.concat([data, data2])

    # Delete all duplicate rows
    data = data.drop_duplicates()

    # Delete all columns except "TS_Samples", "Clusters" and "RMSE_Clusters"
    data = data[['TS_Samples', 'Clusters', 'MAE_Clusters', 'MAE_Intermediate', 'Timestamp']]

    # Delete rows with TS_Samples = 10
    data = data[data.TS_Samples != 10]

    # Sort by lowest RMSE
    data = data.sort_values(by=['MAE_Clusters'])

    # Save the first row for every unique combination of "TS_Samples" and "Clusters" and delete the rest
    data = data.drop_duplicates(subset=['TS_Samples', 'Clusters'])

    # Group data by 'TS_Samples'
    grouped_data = data.groupby('TS_Samples')

    #Express "MAE_Clusters" and "MAE_Intermediate" as percentages of the mean (472 half minutes)
    data['MAE_Clusters'] = data['MAE_Clusters'] / 472 * 100
    data['MAE_Intermediate'] = data['MAE_Intermediate'] / 472 * 100

    # Define colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_data)))

    # Plot each group separately
    for i, (name, group) in enumerate(grouped_data):
        # Sort the data within the group based on 'Clusters' column
        sorted_group = group.sort_values(by='Clusters')
        # Plot scatter points and lines for "RMSE_Clusters"
        plt.scatter(sorted_group['Clusters'], sorted_group['MAE_Clusters'], label=f"TS_Samples = {name}", color=colors[i])
        plt.plot(sorted_group['Clusters'], sorted_group['MAE_Clusters'], linestyle='-', color=colors[i])

    plt.xlabel('Clusters', fontsize=14)
    plt.ylabel('Error (%) [MAE]', fontsize=14)
    plt.title('Error in percentages for different Clusters and TS_Samples', fontsize=16)
    plt.xticks(np.arange(2, 21, step=1), fontsize=14)  # Set x-axis ticks from 2 to 20
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

error_percentage()
