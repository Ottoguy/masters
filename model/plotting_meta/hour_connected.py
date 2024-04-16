import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from datetime import datetime
from matplotlib.colors import ListedColormap

def HourConnected():
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

    #greys not really used because the is no floor=0, hence this is used in hour_connected_tot.py
    colours = [plt.cm.Greys, plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]

    # Find the floor with the maximum count
    max_count_floor = data.groupby('Floor')['Half_Minutes'].count().idxmax()

    # Calculate the maximum count for normalizing the color, based on the floor with the maximum count
    max_count = data[data['Floor'] == max_count_floor].groupby('HourConnected')['Half_Minutes'].count().max()

    #Loop through the unique values of "Floor" column
    for floor_value in data['Floor'].unique():
        print(f"Plotting for floor {floor_value}", end='\r')
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Set the title of the figure
        fig.suptitle(f'Floor {floor_value} - TimeConnected and TimeDisconnected', fontsize=16)

        # Function to format y-axis labels as hh:mm
        def format_half_minutes(x, pos):
            return int((x*0.5)/60)

        for ax in axs.flat:
            ax.set_xlim(0, 23)  # Set x-axis limit from 0 to 23 for all subplots
            # Plotting and formatting for each subplot
            if ax == axs[0, 0]:
                data_floor = data[(data['Weekend'] == False) & (data['Floor'] == floor_value)]
                data_weekday_connected = data_floor.groupby('HourConnected')['Half_Minutes'].median()
                count_weekday_connected = data_floor.groupby('HourConnected')['Half_Minutes'].count()
                ax.bar(data_weekday_connected.index, data_weekday_connected, width=0.8, color=colours[floor_value](count_weekday_connected / max_count))
                ax.set_title('Weekend=False - TimeConnected')
                ax.set_xlabel('Hour of the Day')
                ax.set_ylabel('Median Charging Duration (hours)')
                ax.set_xticks(data_weekday_connected.index)
            elif ax == axs[0, 1]:
                data_floor = data[(data['Weekend'] == True) & (data['Floor'] == floor_value)]
                data_weekend_connected = data_floor.groupby('HourConnected')['Half_Minutes'].median()
                count_weekend_connected = data_floor.groupby('HourConnected')['Half_Minutes'].count()
                ax.bar(data_weekend_connected.index, data_weekend_connected, width=0.8, color=colours[floor_value](count_weekend_connected / max_count))
                ax.set_title('Weekend=True - TimeConnected')
                ax.set_xlabel('Hour of the Day')
                ax.set_ylabel('Median Charging Duration (hours)')
                ax.set_xticks(data_weekend_connected.index)
            elif ax == axs[1, 0]:
                data_floor = data[(data['Weekend'] == False) & (data['Floor'] == floor_value)]
                data_weekday_disconnected = data_floor.groupby('HourDisconnected')['Half_Minutes'].median()
                count_weekday_disconnected = data_floor.groupby('HourDisconnected')['Half_Minutes'].count()
                ax.bar(data_weekday_disconnected.index, data_weekday_disconnected, width=0.8, color=colours[floor_value](count_weekday_disconnected / max_count))
                ax.set_title('Weekend=False - TimeDisconnected')
                ax.set_xlabel('Hour of the Day')
                ax.set_ylabel('Median Charging Duration (hours)')
                ax.set_xticks(data_weekday_disconnected.index)
            elif ax == axs[1, 1]:
                data_floor = data[(data['Weekend'] == True) & (data['Floor'] == floor_value)]
                data_weekend_disconnected = data_floor.groupby('HourDisconnected')['Half_Minutes'].median()
                count_weekend_disconnected = data_floor.groupby('HourDisconnected')['Half_Minutes'].count()
                ax.bar(data_weekend_disconnected.index, data_weekend_disconnected, width=0.8, color=colours[floor_value](count_weekend_disconnected / max_count))
                ax.set_title('Weekend=True - TimeDisconnected')
                ax.set_xlabel('Hour of the Day')
                ax.set_ylabel('Median Charging Duration (hours)')
                ax.set_xticks(data_weekend_disconnected.index)

            ax.yaxis.set_major_formatter(FuncFormatter(format_half_minutes))

            # Add a colorbar to each subplot
            norm = Normalize(vmin=0, vmax=max_count)
            sm = ScalarMappable(cmap=colours[floor_value], norm=norm)
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