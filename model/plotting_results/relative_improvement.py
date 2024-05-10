import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def relative_improvement():# Specify the directory where your files are located
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

    # Delete all columns except "TS_Samples", "Clusters" and "MAE_Clusters"
    data = data[['TS_Samples', 'Clusters', 'MAE_Clusters', 'MAE_Intermediate']]

    # Delete rows with TS_Samples = 10
    data = data[data.TS_Samples != 10]

    #Remove rows with NaN values in 'TS_Samples' column
    data = data.dropna(subset=['TS_Samples'])

    # Sort by lowest MAE_Intermediate within each unique combination of TS_Samples
    data_intermediate = data.sort_values(by=['TS_Samples', 'Clusters', 'MAE_Intermediate'])

    # Save the first row for every unique TS_Samples based on MAE_Intermediate
    best_intermediate = data_intermediate.drop_duplicates(subset=['TS_Samples', 'Clusters'])

    # Sort by lowest MAE_Clusters within each unique combination of Clusters and TS_Samples
    data_clusters = data.sort_values(by=['TS_Samples', 'Clusters', 'MAE_Clusters'])

    # Save the first row for every unique combination of Clusters and TS_Samples based on MAE_Clusters
    best_clusters = data_clusters.drop_duplicates(subset=['TS_Samples', 'Clusters'])

    # Merge the best_intermediate with the best_clusters dataframes on 'TS_Samples' and 'Clusters'
    merged_data = best_intermediate.merge(best_clusters, on=['TS_Samples', 'Clusters'], suffixes=('_intermediate', '_clusters'))

    #Delete the columns 'MAE_Intermediate_clusters' and 'MAE_Clusters_intermediate' from the merged dataframe
    merged_data = merged_data.drop(columns=['MAE_Intermediate_clusters', 'MAE_Clusters_intermediate'])

    #Rename the columns MAE_Intermediate_intermediate and MAE_Clusters_clusters to MAE_Intermediate and MAE_Clusters
    merged_data = merged_data.rename(columns={'MAE_Intermediate_intermediate': 'MAE_Intermediate', 'MAE_Clusters_clusters': 'MAE_Clusters'})

    # Make a new column with the percentage decrease in error from intermediate to clusters
    merged_data['Percentage_Decrease'] = ((merged_data['MAE_Intermediate'] - merged_data['MAE_Clusters']) / merged_data['MAE_Intermediate']) * 100

    print(merged_data)

    # Plotting
    unique_samples = merged_data['TS_Samples'].unique()
    for sample in unique_samples:
        sample_data = merged_data[merged_data['TS_Samples'] == sample]
        plt.plot(sample_data['Clusters'], sample_data['Percentage_Decrease'], marker='o', label=f'TS_Samples = {sample}', alpha=1.0)

    plt.xlabel('Clusters', fontsize=14)
    plt.ylabel('Percentage decrease in MAE', fontsize=14)
    plt.title('Percentage decrease in MAE when using clusters', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(which='both', linestyle=':', linewidth=0.5)  # Set grid for both major and minor ticks
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))  # Set y-axis ticks to include percent sign
    plt.xticks(np.arange(2, 21, 1))  # Set x-axis ticks to include integers from 2 to 20
    #increase font size of x and y ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

relative_improvement()