import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def predictions_plot():
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

    # Delete all duplicate rows
    data = data.drop_duplicates()

    #Delete rows with 'TS_Samples' = NaN
    data = data.dropna(subset=['TS_Samples'])

    #Sort by lowest RMSE
    data = data.sort_values(by=['RMSE_Clusters'])

    # Delete all irrelevant columns
    data = data[['TS_Samples', 'Clusters', 'RMSE_Clusters', 'RMSE_Intermediate', 'MAE_Clusters', 'MAE_Intermediate', 'Timestamp']]

    # Delete rows with TS_Samples = 10
    data = data[data.TS_Samples != 10]

    #Delete all rows except every unique value of "TS_Samples" with the lowest "RMSE_Clusters"
    data = data.drop_duplicates(subset=['TS_Samples'], keep='first')

    print(data)

    #Make an array of the 'Timestamp' column
    timestamps = data['Timestamp'].to_numpy()

    #Make an array of the 'Clusters' column
    clusters = data['Clusters'].to_numpy()

    ts_samples = data['TS_Samples'].to_numpy()

    # Specify the directory where your files are located
    folder_path = 'prints/deep_learning/'

    # Create a pattern to match files in the specified format
    file_pattern = '*'

    # Get a list of all files matching the pattern
    file_list = glob.glob(os.path.join(folder_path, file_pattern))

    num_cols = 2
    num_rows = 2

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    axs = axs.flatten()

    # Load and plot data for each timestamp and cluster value
    for i in range(len(timestamps)):
        timestamp = timestamps[i]
        cluster = clusters[i]
        ts_sample = ts_samples[i]

        #Convert to integer
        ts_sample = int(ts_sample)
        cluster = int(cluster)

        # Create a pattern to match files in the specified format
        file_pattern = f"{timestamp}_samples_{ts_sample}_*"

        # Get a list of all files matching the pattern
        matching_files = glob.glob(os.path.join(folder_path, file_pattern))

        # Load the data from the file
        prediction_data = pd.read_csv(matching_files[0])

        # Plot on the appropriate subplot
        ax = axs[i]
        ax.plot(prediction_data.index, prediction_data['Real'], label='Real')
        ax.plot(prediction_data.index, prediction_data['Intermediate'], label='Intermediate')
        ax.plot(prediction_data.index, prediction_data['Clusters'], label='Clusters')

        # Set labels and title
        ax.set_xlabel('Entries')
        ax.set_ylabel('Values')
        ax.set_title(f'TS_Samples: {ts_sample}, Clusters: {cluster}, Timestamp: {timestamp}')

        # Add legend
        ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()




predictions_plot()