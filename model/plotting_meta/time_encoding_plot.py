import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime

def TimeEncodingPlot():
    # Your existing code to load data
    folder_path = 'prints/meta/'
    file_pattern = '*'
    file_list = glob.glob(os.path.join(folder_path, file_pattern))
    file_list.sort(key=os.path.getmtime, reverse=True)
    latest_file = file_list[0]
    data = pd.read_csv(latest_file)

    # Convert the 'TimeConnected' and 'TimeDisconnected' columns to datetime objects
    data['TimeConnected'] = pd.to_datetime(data['TimeConnected'])
    data['TimeDisconnected'] = pd.to_datetime(data['TimeDisconnected'])

    # Extract hour and minute from 'TimeConnected' and 'TimeDisconnected'
    data['TimeConnected_hour'] = data['TimeConnected'].dt.hour
    data['TimeConnected_minute'] = data['TimeConnected'].dt.minute
    data['TimeDisconnected_hour'] = data['TimeDisconnected'].dt.hour
    data['TimeDisconnected_minute'] = data['TimeDisconnected'].dt.minute

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot for TimeConnected
    axs[0].scatter(data['TimeConnected_hour'] + data['TimeConnected_minute']/60, data['TimeConnected_sin'], label='TimeConnected_sin')
    axs[0].scatter(data['TimeConnected_hour'] + data['TimeConnected_minute']/60, data['TimeConnected_cos'], label='TimeConnected_cos')
    axs[0].set_title('TimeConnected_sin and TimeConnected_cos')
    axs[0].set_xlabel('Time of Day (hh:mm - TimeConnected)')
    axs[0].set_ylabel('Values')
    axs[0].legend()

    # Plot for TimeDisconnected
    axs[1].scatter(data['TimeDisconnected_hour'] + data['TimeDisconnected_minute']/60, data['TimeDisconnected_sin'], label='TimeDisconnected_sin')
    axs[1].scatter(data['TimeDisconnected_hour'] + data['TimeDisconnected_minute']/60, data['TimeDisconnected_cos'], label='TimeDisconnected_cos')
    axs[1].set_title('TimeDisconnected_sin and TimeDisconnected_cos')
    axs[1].set_xlabel('Time of Day (hh:mm - TimeDisconnected)')
    axs[1].set_ylabel('Values')
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure with the current date and time in the filename
    results_dir = "plots/time_encoding_plot"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
    plt.close()
