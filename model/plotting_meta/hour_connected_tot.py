import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from datetime import datetime

def HourConnectedTot():
    # Specify the directory where your files are located
    folder_path = 'prints/meta/'

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

    # Convert 'TimeConnected' and 'TimeDisconnected' to datetime format
    data['TimeConnected'] = pd.to_datetime(data['TimeConnected'], format='%Y-%m-%d-%H:%M:%S.%f')
    data['TimeDisconnected'] = pd.to_datetime(data['TimeDisconnected'], format='%Y-%m-%d-%H:%M:%S.%f')

    # Extract the hour information from 'TimeConnected' and 'TimeDisconnected'
    data['HourConnected'] = data['TimeConnected'].dt.hour
    data['HourDisconnected'] = data['TimeDisconnected'].dt.hour

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Function to format y-axis labels as hh:mm
    def format_half_minutes(x, pos):
        return int((x*0.5)/60)

    # Get the maximum count for normalizing the color
    max_count = data.groupby('HourConnected')['Half_Minutes'].count().max()

    #Set the title of the figure
    fig.suptitle('TimeConnected and TimeDisconnected – All floors', fontsize=16)

    for ax in axs.flat:
        # Plotting and formatting for each subplot
        if ax == axs[0, 0]:
            data_weekday_connected = data[(data['Weekend'] == False)].groupby('HourConnected')['Half_Minutes'].median()
            count_weekday_connected = data[(data['Weekend'] == False)].groupby('HourConnected')['Half_Minutes'].count()
            ax.bar(data_weekday_connected.index, data_weekday_connected, width=0.8, color=plt.cm.Greens(count_weekday_connected / max_count))
            ax.set_title('Weekend=False - TimeConnected')
            ax.set_xlabel('Hour of the Day')
            ax.set_ylabel('Median Charging Duration (hours)')
            ax.set_xticks(data_weekday_connected.index)
        elif ax == axs[0, 1]:
            data_weekend_connected = data[data['Weekend'] == True].groupby('HourConnected')['Half_Minutes'].median()
            count_weekend_connected = data[data['Weekend'] == True].groupby('HourConnected')['Half_Minutes'].count()
            ax.bar(data_weekend_connected.index, data_weekend_connected, width=0.8, color=plt.cm.Greens(count_weekend_connected / max_count))
            ax.set_title('Weekend=True - TimeConnected')
            ax.set_xlabel('Hour of the Day')
            ax.set_ylabel('Median Charging Duration (hours)')
            ax.set_xticks(data_weekend_connected.index)
        elif ax == axs[1, 0]:
            data_weekday_disconnected = data[data['Weekend'] == False].groupby('HourDisconnected')['Half_Minutes'].median()
            count_weekday_disconnected = data[data['Weekend'] == False].groupby('HourDisconnected')['Half_Minutes'].count()
            ax.bar(data_weekday_disconnected.index, data_weekday_disconnected, width=0.8, color=plt.cm.Greens(count_weekday_disconnected / max_count))
            ax.set_title('Weekend=False - TimeDisconnected')
            ax.set_xlabel('Hour of the Day')
            ax.set_ylabel('Median Charging Duration (hours)')
            ax.set_xticks(data_weekday_disconnected.index)
        elif ax == axs[1, 1]:
            data_weekend_disconnected = data[data['Weekend'] == True].groupby('HourDisconnected')['Half_Minutes'].median()
            count_weekend_disconnected = data[data['Weekend'] == True].groupby('HourDisconnected')['Half_Minutes'].count()
            ax.bar(data_weekend_disconnected.index, data_weekend_disconnected, width=0.8, color=plt.cm.Greens(count_weekend_disconnected / max_count))
            ax.set_title('Weekend=True - TimeDisconnected')
            ax.set_xlabel('Hour of the Day')
            ax.set_ylabel('Median Charging Duration (hours)')
            ax.set_xticks(data_weekend_disconnected.index)

        ax.yaxis.set_major_formatter(FuncFormatter(format_half_minutes))

        # Add a colorbar to each subplot
        norm = Normalize(vmin=0, vmax=max_count)
        sm = ScalarMappable(cmap=plt.cm.Greens, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Number of Entries')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    results_dir = "plots/hour_connected/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the figure with the current date and time in the filename
    print(f"Saving figure {current_datetime}", end='\r')
    plt.savefig(os.path.join(results_dir, current_datetime + '.png'))

    plt.close()