import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def relative_improvement():# Specify the directory where your files are located
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

    # Sort by lowest RMSE_Intermediate within each unique combination of TS_Samples
    data_intermediate = data.sort_values(by=['TS_Samples', 'RMSE_Intermediate'])

    # Save the first row for every unique TS_Samples based on RMSE_Intermediate
    best_intermediate = data_intermediate.drop_duplicates(subset=['TS_Samples'])

    # Sort by lowest RMSE_Clusters within each unique combination of Clusters and TS_Samples
    data_clusters = data.sort_values(by=['TS_Samples', 'Clusters', 'RMSE_Clusters'])

    # Save the first row for every unique combination of Clusters and TS_Samples based on RMSE_Clusters
    best_clusters = data_clusters.drop_duplicates(subset=['TS_Samples', 'Clusters'])

    # Merge the best_intermediate with the best_clusters dataframes on 'TS_Samples'
    merged_data = best_intermediate.merge(best_clusters, on=['TS_Samples'], suffixes=('_intermediate', '_clusters'))

    #Delete the columns 'RMSE_Intermediate_clusters' and 'RMSE_Clusters_intermediate' from the merged dataframe
    merged_data = merged_data.drop(columns=['RMSE_Intermediate_clusters', 'RMSE_Clusters_intermediate'])

    #Rename the columns RMSE_Intermediate_intermediate and RMSE_Clusters_clusters to RMSE_Intermediate and RMSE_Clusters
    merged_data = merged_data.rename(columns={'RMSE_Intermediate_intermediate': 'RMSE_Intermediate', 'RMSE_Clusters_clusters': 'RMSE_Clusters'})

    # Make a new column with the percentage decrease in error from intermediate to clusters
    merged_data['Percentage_Decrease'] = ((merged_data['RMSE_Intermediate'] - merged_data['RMSE_Clusters']) / merged_data['RMSE_Intermediate']) * 100

    print(merged_data)

    # Plotting
    unique_samples = merged_data['TS_Samples'].unique()
    for sample in unique_samples:
        sample_data = merged_data[merged_data['TS_Samples'] == sample]
        plt.plot(sample_data['Clusters'], sample_data['Percentage_Decrease'], marker='o', label=f'TS_Samples = {sample}', alpha=1.0)

    plt.xlabel('Clusters')
    plt.ylabel('Percentage decrease in error')
    plt.title('Percentage decrease in error when using clusters')
    plt.legend()
    plt.grid(which='both', linestyle=':', linewidth=0.5)  # Set grid for both major and minor ticks
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))  # Set y-axis ticks to include percent sign
    plt.xticks(np.arange(2, 21, 1))  # Set x-axis ticks to include integers from 2 to 20
    plt.show()

relative_improvement()